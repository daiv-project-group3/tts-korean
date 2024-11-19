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
        # 마스크 생성
        mask = ~self.get_mask_from_lengths(mel_lengths)
        mask = mask.expand(mel_target.size(2), mask.size(0), mask.size(1))
        mask = mask.permute(1, 2, 0)
        
        # 마스킹 적용
        mel_target.masked_fill_(mask, 0)
        mel_output.masked_fill_(mask, 0)
        mel_output_postnet.masked_fill_(mask, 0)
        
        # Loss 계산
        mel_loss = self.l1_loss(mel_output, mel_target) + \
                  self.l1_loss(mel_output_postnet, mel_target)
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

    def feature_matching_loss(self, real_feats, fake_feats):
        """Feature Matching Loss for both MSD and MPD"""
        total_fm_loss = 0.0
        
        # Multi-scale discriminator feature matching
        for real_feat_list, fake_feat_list in zip(real_feats['msd'], fake_feats['msd']):
            for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
                total_fm_loss += self.l1_loss(fake_feat, real_feat.detach())
        
        # Multi-period discriminator feature matching
        for real_feat_list, fake_feat_list in zip(real_feats['mpd'], fake_feats['mpd']):
            for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
                total_fm_loss += self.l1_loss(fake_feat, real_feat.detach())
        
        return total_fm_loss * self.lambda_fm

    def generator_loss(self, disc_outputs):
        """Generator Adversarial Loss"""
        total_gen_loss = 0.0
        
        # MSD losses
        for msd_output in disc_outputs['msd']:
            total_gen_loss += self.mse_loss(msd_output, 
                                          torch.ones_like(msd_output))
            
        # MPD losses
        for mpd_output in disc_outputs['mpd']:
            total_gen_loss += self.mse_loss(mpd_output, 
                                          torch.ones_like(mpd_output))
            
        return total_gen_loss * self.lambda_adv

    def discriminator_loss(self, real_outputs, fake_outputs):
        """Discriminator Loss"""
        total_disc_loss = 0.0
        
        # MSD losses
        for real_msd, fake_msd in zip(real_outputs['msd'], fake_outputs['msd']):
            real_loss = self.mse_loss(real_msd, torch.ones_like(real_msd))
            fake_loss = self.mse_loss(fake_msd, torch.zeros_like(fake_msd))
            total_disc_loss += (real_loss + fake_loss) * 0.5
            
        # MPD losses
        for real_mpd, fake_mpd in zip(real_outputs['mpd'], fake_outputs['mpd']):
            real_loss = self.mse_loss(real_mpd, torch.ones_like(real_mpd))
            fake_loss = self.mse_loss(fake_mpd, torch.zeros_like(fake_mpd))
            total_disc_loss += (real_loss + fake_loss) * 0.5
            
        return total_disc_loss * self.lambda_adv

    def hifi_gan_loss(self, real_wave, fake_wave, real_outputs, fake_outputs, real_feats, fake_feats):
        """HiFi-GAN Total Loss Calculation"""
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