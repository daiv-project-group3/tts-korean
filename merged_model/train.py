import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataset import KSSTTSDataset
from tacotron2 import Tacotron2
from hifi_gan import Generator, MultiScaleDiscriminator, MultiPeriodDiscriminator, MRFDiscriminator
from loss import TTSLoss

class Trainer:
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(config.device_num)
            self.device = torch.device("cuda")
        
        # 데이터셋 및 데이터로더 초기화
        self.train_dataset = KSSTTSDataset(split='train')
        self.valid_dataset = KSSTTSDataset(split='valid')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=KSSTTSDataset.collate_fn,
            num_workers=4
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=KSSTTSDataset.collate_fn,
            num_workers=4
        )
        
        # 모델 초기화
        self.tacotron2 = Tacotron2(
            n_mel_channels=80,
            vocab_size=len(self.train_dataset.vocab),
            embedding_dim=256,
            encoder_n_convolutions=3,
            encoder_kernel_size=5,
            attention_rnn_dim=512,
            attention_dim=128
        ).to(self.device)
        
        self.generator = Generator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.mrf = MRFDiscriminator().to(self.device)
        
        # Loss 함수 초기화
        self.criterion = TTSLoss(device=self.device)
        
        # Optimizer 초기화
        self.optimizer_t2 = optim.Adam(self.tacotron2.parameters(), lr=config.learning_rate, weight_decay=1e-6)
        self.optimizer_g = optim.Adam(self.generator.parameters(), 
                                    lr=config.learning_rate,
                                    weight_decay=1e-6,
                                    betas=(0.8, 0.99))
        self.optimizer_d = optim.Adam(
            list(self.msd.parameters()) + 
            list(self.mpd.parameters()) + 
            list(self.mrf.parameters()),
            lr=config.learning_rate,
            weight_decay=1e-6,
            betas=(0.8, 0.99)
        )
        
        # 체크포인트 디렉토리 생성
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'tacotron2_state_dict': self.tacotron2.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'msd_state_dict': self.msd.state_dict(),
            'mpd_state_dict': self.mpd.state_dict(),
            'mrf_state_dict': self.mrf.state_dict(),
            'optimizer_t2_state_dict': self.optimizer_t2.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'loss': loss
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        self.tacotron2.load_state_dict(checkpoint['tacotron2_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.msd.load_state_dict(checkpoint['msd_state_dict'])
        self.mpd.load_state_dict(checkpoint['mpd_state_dict'])
        self.mrf.load_state_dict(checkpoint['mrf_state_dict'])
        
        self.optimizer_t2.load_state_dict(checkpoint['optimizer_t2_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        return checkpoint['epoch']

    def train_step(self, batch):
        # 변수 초기화
        mel_outputs = None
        mel_outputs_postnet = None
        fake_audio = None
        
        try:
            # 데이터 로드 및 디바이스 이동
            text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, audio_padded = [x.to(self.device) for x in batch]
            
            # 1. Tacotron2 forward pass
            mel_outputs, mel_outputs_postnet, gate_outputs, _ = self.tacotron2(
                text_padded, input_lengths, mel_padded, output_lengths)
            
            # mel_outputs의 차원을 확인하고 조정
            B, T, C = mel_outputs.size()
            if C != 80:  # n_mel_channels가 80이 아닌 경우
                mel_outputs = mel_outputs[:, :, :80]
                mel_outputs_postnet = mel_outputs_postnet[:, :, :80]
            
            # Tacotron2 loss 계산
            t2_losses = self.criterion.tacotron2_loss(
                mel_outputs, mel_outputs_postnet, gate_outputs,
                mel_padded, gate_padded, output_lengths
            )
            
            # 2. Generator forward pass
            mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)  # [B, T, C] -> [B, C, T]
            fake_audio = self.generator(mel_outputs_postnet)
            
            # 오디오 차원 확인 및 조정
            if fake_audio.dim() == 2:
                fake_audio = fake_audio.unsqueeze(1)
            if audio_padded.dim() == 2:
                audio_padded = audio_padded.unsqueeze(1)
            
            # Discriminator forward passes
            msd_real_outputs, msd_real_features = self.msd(audio_padded)
            msd_fake_outputs, msd_fake_features = self.msd(fake_audio.detach())
            
            mpd_real_outputs, mpd_real_features = self.mpd(audio_padded)
            mpd_fake_outputs, mpd_fake_features = self.mpd(fake_audio.detach())
            
            # MRF Discriminator
            mrf_output = self.mrf(msd_fake_features, mpd_fake_features)
            
            # HiFi-GAN losses
            hifigan_losses = self.criterion.hifi_gan_loss(
                audio_padded, fake_audio,
                msd_real_outputs + mpd_real_outputs,  # real outputs
                msd_fake_outputs + mpd_fake_outputs,  # fake outputs
                msd_real_features + mpd_real_features,  # real features
                msd_fake_features + mpd_fake_features   # fake features
            )
            
            # 5. Optimization
            # Tacotron2
            self.optimizer_t2.zero_grad()
            t2_losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.tacotron2.parameters(), 1.0)
            self.optimizer_t2.step()
            
            # Generator
            self.optimizer_g.zero_grad()
            hifigan_losses['generator_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.optimizer_g.step()
            
            # Discriminators
            self.optimizer_d.zero_grad()
            hifigan_losses['discriminator_loss'].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.msd.parameters()) + 
                list(self.mpd.parameters()) + 
                list(self.mrf.parameters()),
                1.0
            )
            self.optimizer_d.step()
            
            return {**t2_losses, **hifigan_losses}
            
        except Exception as e:
            print(f"\nError in train_step: {str(e)}")
            print("\nShape information:")
            print(f"text_padded: {text_padded.shape}")
            print(f"mel_padded: {mel_padded.shape} (expected: [B, n_mel_channels, T])")
            if mel_outputs is not None:
                print(f"mel_outputs: {mel_outputs.shape} (expected: [B, T, n_mel_channels])")
                print(f"gate_outputs: {gate_outputs.shape} (expected: [B, T])")
            if mel_outputs_postnet is not None:
                print(f"mel_outputs_postnet: {mel_outputs_postnet.shape} (expected: [B, T, n_mel_channels])")
            if fake_audio is not None:
                print(f"fake_audio: {fake_audio.shape}")
            print(f"audio_padded: {audio_padded.shape}")
            print(f"gate_padded: {gate_padded.shape} (expected: [B, T])")
            raise e

    def train(self):
        start_epoch = 0
        if self.config.resume_checkpoint:
            start_epoch = self.load_checkpoint(self.config.resume_checkpoint)
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.tacotron2.train()
            self.generator.train()
            self.msd.train()
            self.mpd.train()
            self.mrf.train()
            
            total_loss = 0
            progress_bar = tqdm(self.train_loader)
            
            for batch in progress_bar:
                losses = self.train_step(batch)
                total_loss += losses['total_loss'].item()
                
                progress_bar.set_description(
                    f"Epoch {epoch+1}, Loss: {losses['total_loss'].item():.4f}"
                )
            
            avg_loss = total_loss / len(self.train_loader)
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1, avg_loss)
            
            print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    from argparse import Namespace
    
    config = Namespace(
        device_num=0,
        batch_size=32,
        learning_rate=0.0002,
        num_epochs=100,
        start_epoch=0,
        save_interval=10,
        checkpoint_dir='checkpoints',
        resume_checkpoint=None
    )
    
    trainer = Trainer(config)
    trainer.train() 