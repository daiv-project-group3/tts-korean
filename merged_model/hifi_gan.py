import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                                padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                padding=get_padding(kernel_size, 1)))
            for _ in range(3)
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class Generator(nn.Module):
    def __init__(self, input_size=80):
        super().__init__()
        self.input_size = input_size
        
        # Initial conv layer with matching input channels
        self.conv_pre = weight_norm(nn.Conv1d(input_size, 256, 7, 1, 3))
        
        # Upsampling layers with proper channel sizes
        self.ups = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(256, 128, 16, 8, 4)),
            weight_norm(nn.ConvTranspose1d(128, 64, 16, 8, 4)),
            weight_norm(nn.ConvTranspose1d(64, 32, 4, 2, 1)),
            weight_norm(nn.ConvTranspose1d(32, 16, 4, 2, 1)),
        ])
        
        # Residual blocks with matching channel sizes
        self.resblocks = nn.ModuleList([
            nn.ModuleList([
                ResBlock(128, kernel_size=3, dilation=(1, 3, 5))
                for _ in range(3)
            ]),
            nn.ModuleList([
                ResBlock(64, kernel_size=3, dilation=(1, 3, 5))
                for _ in range(2)
            ]),
            nn.ModuleList([
                ResBlock(32, kernel_size=3, dilation=(1, 3, 5))
                for _ in range(2)
            ]),
            nn.ModuleList([
                ResBlock(16, kernel_size=3, dilation=(1, 3, 5))
            ]),
        ])
        
        # Final conv layer
        self.conv_post = weight_norm(nn.Conv1d(16, 1, 7, 1, 3))
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Initial conv
        x = self.conv_pre(x)
        
        # Upsampling and residual blocks
        for up, res_blocks in zip(self.ups, self.resblocks):
            x = up(x)
            x = F.leaky_relu(x, 0.1)
            for res in res_blocks:
                x = res(x)
        
        # Final conv
        x = self.conv_post(x)
        x = self.tanh(x)
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

    def forward(self, y, y_hat):
        y_d_rs = []  # real outputs
        y_d_gs = []  # generated outputs
        fmap_rs = []  # real features
        fmap_gs = []  # generated features

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

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

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class MRFDiscriminator(nn.Module):
    def __init__(self, resolutions=[(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400)]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MRFSubDiscriminator(resolution) for resolution in resolutions
        ])

    def forward(self, y, y_hat):
        real_outputs = []
        fake_outputs = []
        real_fmaps = []
        fake_fmaps = []

        for disc in self.discriminators:
            r_out, r_fmap = disc(y)
            f_out, f_fmap = disc(y_hat)
            
            real_outputs.append(r_out)
            fake_outputs.append(f_out)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)

        return real_outputs, fake_outputs, real_fmaps, fake_fmaps

class MRFSubDiscriminator(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution
        kernel_size, stride, padding = resolution
        
        # FDCNN (Frequency-Discriminative Convolutional Neural Network)
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 32, kernel_size, stride=stride, padding=padding)),
            weight_norm(nn.Conv1d(32, 64, kernel_size=41, stride=4, padding=20, groups=4)),
            weight_norm(nn.Conv1d(64, 128, kernel_size=41, stride=4, padding=20, groups=16)),
            weight_norm(nn.Conv1d(128, 256, kernel_size=41, stride=4, padding=20, groups=16)),
            weight_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20, groups=32)),
            weight_norm(nn.Conv1d(512, 1024, kernel_size=41, stride=4, padding=20, groups=32)),
            weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
            weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
        ])
        
        self.activations = nn.ModuleList([
            nn.LeakyReLU(0.2, True) for _ in range(len(self.layers) - 1)
        ])

    def forward(self, x):
        feature_maps = []
        
        for i, (layer, activation) in enumerate(zip(self.layers[:-1], self.activations)):
            x = layer(x)
            x = activation(x)
            feature_maps.append(x)
        
        x = self.layers[-1](x)
        
        return x, feature_maps

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

class DiscriminatorR(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 3, 1, padding=1)),
            weight_norm(nn.Conv1d(128, 256, 3, 2, padding=1)),
            weight_norm(nn.Conv1d(256, 512, 3, 2, padding=1)),
            weight_norm(nn.Conv1d(512, 1024, 3, 2, padding=1)),
            weight_norm(nn.Conv1d(1024, 1024, 3, 1, padding=1)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

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

class MultiResolutionFourierDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorR(),
            DiscriminatorR(),
            DiscriminatorR(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs