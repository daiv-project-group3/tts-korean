import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, negative_slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip_connection(x)
        x = self.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        return self.leaky_relu(x + residual)

class Generator(nn.Module):
    def __init__(self, input_size=80):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 512, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU()

        # 첫 번째 업샘플링 레이어
        self.deconv1 = nn.ConvTranspose1d(512, 512, kernel_size=10, stride=2, padding=4)

        # 두 번째 업샘플링 레이어
        self.deconv2 = nn.ConvTranspose1d(512, 1, kernel_size=10, stride=2, padding=4)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.deconv1(x))
        x = self.deconv2(x)
        return self.tanh(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(4, 16, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
        ])
        self.final_conv = nn.Conv1d(128, 1, 3, 1, 1)

        # real_audio 텐서를 4개의 채널로 변환하는 Conv1d 추가
        self.channel_conv = nn.Conv1d(4, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # real_audio의 채널을 4로 변환
        x = self.channel_conv(x)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        x = self.final_conv(x)
        return x, features

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(4, 16, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=8),
            nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=16),
            nn.Conv1d(64, 128, kernel_size=64, stride=8, padding=32),
        ])
        self.final_conv = nn.Conv1d(128, 1, 3, 1, 1)
        
        self.channel_conv = nn.Conv1d(4, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.channel_conv(x)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        x = self.final_conv(x)
        return x, features

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
        # MSD와 MPD에서 나온 특징을 결합
        fused_features = torch.cat([msd_features[-1], mpd_features[-1]], dim=1)
        
        # 채널 수를 맞추기 위해 adjustment
        fused_features = self.channel_adjustment(fused_features)
        
        # ResBlock을 통한 특성 추출
        x = self.resblock1(fused_features)
        x = self.resblock2(x)
        
        # Fusion layer
        x = self.fusion_layer(x)
        x = self.output_layer(x)
        return x