import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class TTSLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Loss weights from the paper
        self.lambda_mel = 45.0
        self.lambda_fm = 2.0
        self.lambda_adv = 1.0
        
        # Mel-spectrogram transform configuration
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0,
            f_max=8000,
            n_mels=80,
            power=1.0,
            normalized=True
        ).to(device)

    def tacotron2_loss(self, mel_output, mel_output_postnet, gate_out, 
                      mel_target, gate_target, mel_lengths):
        """Tacotron2 Loss Calculation"""
        # mel_target: [B, n_mel_channels, T]
        # mel_output: [B, T, n_mel_channels]
        
        # 차원 맞추기
        mel_target = mel_target.transpose(1, 2)  # [B, T, n_mel_channels]
        
        # gate_out 차원 조정 [B, T, 1] -> [B, T]
        gate_out = gate_out.squeeze(-1)
        
        # 텐서 차원 확인
        B, T, C = mel_target.size()  # [batch_size, time_steps, n_mel_channels]
        
        # 마스크 생성 (B, T)
        mask = ~self.get_mask_from_lengths(mel_lengths)
        
        # 마스크를 멜 스펙트로그램 차원에 맞게 조정
        mel_mask = mask.unsqueeze(-1).expand(-1, -1, C)  # [B, T, C]
        
        # 마스킹 적용
        mel_target_masked = mel_target.masked_fill(mel_mask, 0)
        mel_output_masked = mel_output.masked_fill(mel_mask, 0)
        mel_output_postnet_masked = mel_output_postnet.masked_fill(mel_mask, 0)
        
        # gate에 대한 마스킹 적용
        gate_target = gate_target.masked_fill(mask, 0)
        gate_out = gate_out.masked_fill(mask, 0)
        
        # Loss 계산
        mel_loss = self.l1_loss(mel_output_masked, mel_target_masked) + \
                  self.l1_loss(mel_output_postnet_masked, mel_target_masked)
        gate_loss = self.bce_loss(gate_out, gate_target)
        
        return {
            'mel_loss': mel_loss,
            'gate_loss': gate_loss,
            'total_loss': mel_loss + gate_loss
        }

    def mel_spectrogram_loss(self, real_wave, fake_wave):
        """Mel-spectrogram L1 Loss"""
        real_mel = self.mel_transform(real_wave)
        fake_mel = self.mel_transform(fake_wave)
        return self.l1_loss(fake_mel, real_mel) * self.lambda_mel

    def feature_matching_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2

    def generator_loss(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            loss += torch.mean((1-dg)**2)
        return loss

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
        return loss

    def hifi_gan_loss(self, real_wave, fake_wave, real_outputs, fake_outputs, real_feats, fake_feats):
        """HiFi-GAN Total Loss Calculation"""
        # real_wave와 fake_wave의 차원 확인 및 조정
        if real_wave.dim() == 2:
            real_wave = real_wave.unsqueeze(1)
        if fake_wave.dim() == 2:
            fake_wave = fake_wave.unsqueeze(1)
        
        # 1. Mel-spectrogram Loss
        mel_loss = self.mel_spectrogram_loss(real_wave, fake_wave)
        
        # 2. Feature Matching Loss
        fm_loss = self.feature_matching_loss(real_feats, fake_feats)
        
        # 3. Adversarial Losses
        gen_loss = self.generator_loss(fake_outputs)
        disc_loss = self.discriminator_loss(real_outputs, fake_outputs)
        
        # Total losses
        g_loss = mel_loss + fm_loss + gen_loss
        d_loss = disc_loss
        
        return {
            'generator_loss': g_loss,
            'discriminator_loss': d_loss,
            'mel_loss': mel_loss,
            'feature_matching_loss': fm_loss,
            'adversarial_loss_g': gen_loss,
            'adversarial_loss_d': disc_loss
        }

    @staticmethod
    def get_mask_from_lengths(lengths):
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=lengths.device)
        mask = (ids < lengths.unsqueeze(1)).bool()
        return mask