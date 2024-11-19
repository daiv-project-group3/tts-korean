import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, negative_slope=0.01):
        super().__init__()
        # 패딩 계산을 동적으로 수행
        self.padding = (dilation * (kernel_size - 1)) // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.skip_connection = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else
            nn.Identity()
        )

    def forward(self, x):
        residual = self.skip_connection(x)
        x = self.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        
        # 차원이 다른 경우 residual을 조정
        if x.size(-1) != residual.size(-1):
            # 더 작은 크기에 맞춤
            target_size = min(x.size(-1), residual.size(-1))
            x = x[..., :target_size]
            residual = residual[..., :target_size]
            
        return self.leaky_relu(x + residual)

class Generator(nn.Module):
    def __init__(self, input_size=80):
        super().__init__()
        # Initial conv
        self.conv_pre = nn.Conv1d(input_size, 512, 7, 1, 3)
        
        # Upsampling layers with kernel size and stride adjustments
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, 16, 8, 4),
            nn.ConvTranspose1d(256, 128, 16, 8, 4),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
        ])
        
        # Multi-Receptive Field Fusion
        self.mrf_blocks = nn.ModuleList([
            # First MRF block
            nn.ModuleList([
                ResBlock(256, 256, kernel_size=3, dilation=1),
                ResBlock(256, 256, kernel_size=3, dilation=3),
                ResBlock(256, 256, kernel_size=3, dilation=5)
            ]),
            # Second MRF block
            nn.ModuleList([
                ResBlock(128, 128, kernel_size=3, dilation=1),
                ResBlock(128, 128, kernel_size=3, dilation=3),
                ResBlock(128, 128, kernel_size=3, dilation=5)
            ]),
            # Third MRF block
            nn.ModuleList([
                ResBlock(64, 64, kernel_size=3, dilation=1),
                ResBlock(64, 64, kernel_size=3, dilation=3)
            ]),
            # Fourth MRF block
            nn.ModuleList([
                ResBlock(32, 32, kernel_size=3, dilation=1),
                ResBlock(32, 32, kernel_size=3, dilation=2)
            ])
        ])
        
        # Final conv
        self.conv_post = nn.Conv1d(32, 1, 7, 1, 3)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        print(f"Generator input shape: {x.shape}")
        x = self.conv_pre(x)
        print(f"After conv_pre: {x.shape}")
        
        for i in range(len(self.ups)):
            x = self.leaky_relu(x)
            x = self.ups[i](x)
            print(f"After up {i}: {x.shape}")
            
            # Apply MRF blocks
            xs = None
            for j, resblock in enumerate(self.mrf_blocks[i]):
                if xs is None:
                    xs = resblock(x)
                else:
                    # 크기가 다른 경우 처리
                    res_out = resblock(x)
                    if res_out.size(-1) != xs.size(-1):
                        target_size = min(res_out.size(-1), xs.size(-1))
                        xs = xs[..., :target_size]
                        res_out = res_out[..., :target_size]
                    xs += res_out
            x = xs / len(self.mrf_blocks[i])
        
        print(f"Generator output shape: {x.shape}")
        x = self.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x):
        """
        x: 입력 오디오 (fake 또는 real)
        """
        y_d_rs = []  # discriminator outputs
        fmap_rs = []  # feature maps

        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i-1](x)
            y_d_r, fmap_r = d(x)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs

class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, x):
        """
        x: 입력 오디오 (fake 또는 real)
        """
        y_d_rs = []  # discriminator outputs
        fmap_rs = []  # feature maps

        for d in self.discriminators:
            y_d_r, fmap_r = d(x)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs

class MRFDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 채널 수를 맞추기 위한 Conv1d 레이어
        self.channel_adjustment = nn.Conv1d(128, 512, kernel_size=3, padding=1)
        
        # ResBlock 추가
        self.resblock1 = ResBlock(512, 256)
        self.resblock2 = ResBlock(256, 128)
        
        # Fusion layer
        self.fusion_layer = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.output_layer = nn.Conv1d(64, 1, kernel_size=3, padding=1)

    def forward(self, msd_features, mpd_features):
        # 특성 맵의 시간 차원이 다를 수 있으므로 보간 추가
        target_length = min(msd_features[-1].size(-1), mpd_features[-1].size(-1))
        msd_feat = F.interpolate(msd_features[-1], size=target_length, mode='linear')
        mpd_feat = F.interpolate(mpd_features[-1], size=target_length, mode='linear')
        
        fused_features = torch.cat([msd_feat, mpd_feat], dim=1)
        
        # 채��� 수를 맞추기 위해 adjustment
        fused_features = self.channel_adjustment(fused_features)
        
        # ResBlock을 통한 특성 추출
        x = self.resblock1(fused_features)
        x = self.resblock2(x)
        
        # Fusion layer
        x = self.fusion_layer(x)
        x = self.output_layer(x)
        return x

class DiscriminatorP(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        """
        x: [B, 1, T]
        """
        fmap = []
        
        # 1D -> 2D
        b, c, t = x.shape
        if t % self.period != 0:  # 패딩 추가
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = self.leaky_relu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap