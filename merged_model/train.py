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
            torch.cuda.set_device(self.config.device_num)
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
        text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, audio_padded = [x.to(self.device) for x in batch]
        
        # Tacotron2 forward
        mel_outputs, mel_outputs_postnet, gate_outputs, _ = self.tacotron2(
            text_padded, input_lengths, mel_padded, output_lengths)
        
        # Tacotron2 loss
        t2_losses = self.criterion.tacotron2_loss(
            mel_outputs, mel_outputs_postnet, gate_outputs,
            mel_padded, gate_padded, output_lengths)
        
        # Generator forward
        fake_audio = self.generator(mel_outputs_postnet)
        
        # Discriminator forward
        msd_fake_output, msd_fake_features = self.msd(fake_audio)
        mpd_fake_output, mpd_fake_features = self.mpd(fake_audio)
        
        msd_real_output, msd_real_features = self.msd(audio_padded)
        mpd_real_output, mpd_real_features = self.mpd(audio_padded)
        
        # MRF forward
        mrf_fake_output = self.mrf(msd_fake_features, mpd_fake_features)
        mrf_real_output = self.mrf(msd_real_features, mpd_real_features)
        
        # HiFi-GAN loss
        real_outputs = {
            'msd': msd_real_output,
            'mpd': mpd_real_output,
            'mrf': mrf_real_output
        }
        fake_outputs = {
            'msd': msd_fake_output,
            'mpd': mpd_fake_output,
            'mrf': mrf_fake_output
        }
        real_feats = {
            'msd': msd_real_features,
            'mpd': mpd_real_features
        }
        fake_feats = {
            'msd': msd_fake_features,
            'mpd': mpd_fake_features
        }
        
        hifigan_losses = self.criterion.hifi_gan_loss(
            audio_padded, fake_audio, real_outputs, fake_outputs,
            real_feats, fake_feats)
        
        # Optimize
        # 1. Tacotron2
        self.optimizer_t2.zero_grad()
        t2_losses['total_loss'].backward()
        self.optimizer_t2.step()
        
        # 2. Generator
        self.optimizer_g.zero_grad()
        hifigan_losses['generator_loss'].backward()
        self.optimizer_g.step()
        
        # 3. Discriminator
        self.optimizer_d.zero_grad()
        hifigan_losses['discriminator_loss'].backward()
        self.optimizer_d.step()
        
        return {**t2_losses, **hifigan_losses}

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