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

    def get_mask_from_lengths(self, lengths):
        """
        lengths 텐서로부터 마스크를 생성합니다.
        
        Args:
            lengths (Tensor): shape [B] 배치의 각 시퀀스 길이
            
        Returns:
            Tensor: shape [B, T] 마스크 (True = 유효한 위치, False = 패딩 위치)
        """
        max_len = torch.max(lengths).item()
        batch_size = lengths.size(0)
        
        # [B, T] 크기의 인덱스 그리드 생성
        ids = torch.arange(0, max_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # lengths를 [B, 1] 형태로 확장하여 broadcasting 가능하게 함
        mask = ids < lengths.unsqueeze(1)
        
        return mask
        
    def tacotron2_loss(self, mel_out, mel_out_postnet, gate_out, mel_target, gate_target, mel_lengths):
        # mel_target과 mel_out의 shape 확인
        B, T, C = mel_out.size()
        target_T = mel_target.size(2)
        
        # Adjust target length if necessary
        if T > target_T:
            mel_out = mel_out[:, :target_T, :]
            mel_out_postnet = mel_out_postnet[:, :target_T, :]
            gate_out = gate_out[:, :target_T]
        elif T < target_T:
            # Pad outputs if they're shorter than target
            mel_out = F.pad(mel_out, (0, 0, 0, target_T - T))
            mel_out_postnet = F.pad(mel_out_postnet, (0, 0, 0, target_T - T))
            gate_out = F.pad(gate_out, (0, target_T - T))
        
        # Transpose mel_target to match output shape
        mel_target = mel_target.transpose(1, 2)
        
        # Create mask based on lengths
        mask = ~self.get_mask_from_lengths(mel_lengths)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_target.size(-1))
        
        # Apply mask
        mel_target_masked = mel_target.masked_fill(mask, 0)
        mel_out_masked = mel_out.masked_fill(mask, 0)
        mel_out_postnet_masked = mel_out_postnet.masked_fill(mask, 0)
        
        # Calculate losses
        mel_loss = F.mse_loss(mel_out_masked, mel_target_masked) + \
                   F.mse_loss(mel_out_postnet_masked, mel_target_masked)
        
        # Gate loss
        gate_target = gate_target[:, :mel_out.size(1)]
        gate_target = gate_target.view(-1, 1)
        gate_out = gate_out.view(-1, 1)
        gate_loss = F.binary_cross_entropy_with_logits(gate_out, gate_target)
        
        return {
            'mel_loss': mel_loss,
            'gate_loss': gate_loss,
            'total_loss': mel_loss + gate_loss
        }

    def mel_spectrogram_loss(self, real_wave, fake_wave):
        """Mel-spectrogram L1 Loss"""
        # 더 작은 길이에 맞추기
        min_length = min(real_wave.size(-1), fake_wave.size(-1))
        real_wave = real_wave[..., :min_length]
        fake_wave = fake_wave[..., :min_length]
        
        # Mel spectrogram 생성
        real_mel = self.mel_transform(real_wave)
        fake_mel = self.mel_transform(fake_wave)
        
        # 디버깅을 위한 shape 출력
        # print(f"Real wave shape: {real_wave.shape}")
        # print(f"Fake wave shape: {fake_wave.shape}")
        # print(f"Real mel shape: {real_mel.shape}")
        # print(f"Fake mel shape: {fake_mel.shape}")
        
        return self.l1_loss(fake_mel, real_mel) * self.lambda_mel

    def feature_matching_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                if isinstance(rl, torch.Tensor) and isinstance(gl, torch.Tensor):
                    # 더 작은 크기에 맞추기
                    if rl.dim() == gl.dim():
                        min_length = min(rl.size(-1), gl.size(-1))
                        if rl.dim() == 3:  # MSD features
                            rl = rl[..., :min_length]
                            gl = gl[..., :min_length]
                        elif rl.dim() == 4:  # MPD features
                            rl = rl[..., :min_length, :]
                            gl = gl[..., :min_length, :]
                        
                        loss += torch.mean(torch.abs(rl - gl))
        
        return loss * self.lambda_fm

    def generator_loss(self, disc_outputs):
        loss = 0
        for disc_output_group in disc_outputs:
            if isinstance(disc_output_group, (list, tuple)):
                # 리스트인 경우 각 요소에 대해 처리
                for dg in disc_output_group:
                    if isinstance(dg, torch.Tensor):
                        loss += torch.mean((1-dg)**2)
            elif isinstance(disc_output_group, torch.Tensor):
                # 텐서인 경우 직접 처리
                loss += torch.mean((1-disc_output_group)**2)
        return loss

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr_group, dg_group in zip(disc_real_outputs, disc_generated_outputs):
            if isinstance(dr_group, (list, tuple)) and isinstance(dg_group, (list, tuple)):
                # 리스트인 경우 각 요소에 대해 처리
                for dr, dg in zip(dr_group, dg_group):
                    if isinstance(dr, torch.Tensor) and isinstance(dg, torch.Tensor):
                        r_loss = torch.mean((1-dr)**2)
                        g_loss = torch.mean(dg**2)
                        loss += (r_loss + g_loss)
            elif isinstance(dr_group, torch.Tensor) and isinstance(dg_group, torch.Tensor):
                # 텐서인 경우 직접 처리
                r_loss = torch.mean((1-dr_group)**2)
                g_loss = torch.mean(dg_group**2)
                loss += (r_loss + g_loss)
        return loss

    def hifi_gan_loss(self, real_wave, fake_wave, real_outputs, fake_outputs, real_feats, fake_feats):
        # 더 작은 크기의 feature matching
        fm_loss = self.feature_matching_loss(
            real_feats[:2],  # 처음 2개의 discriminator feature만 사용
            fake_feats[:2]
        )
        
        # Loss 계산 최적화
        mel_loss = self.mel_spectrogram_loss(real_wave, fake_wave)
        gen_loss = self.generator_loss(fake_outputs)
        disc_loss = self.discriminator_loss(real_outputs, fake_outputs)
        
        # 메모리 해제
        del real_feats, fake_feats
        torch.cuda.empty_cache()
        
        return {
            'generator_loss': mel_loss + fm_loss + gen_loss,
            'discriminator_loss': disc_loss,
            'mel_loss': mel_loss,
            'feature_matching_loss': fm_loss,
            'adversarial_loss_g': gen_loss,
            'adversarial_loss_d': disc_loss
        }