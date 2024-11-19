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
        self.optimizer_g = optim.AdamW([
            {'params': self.tacotron2.parameters()},
            {'params': self.generator.parameters()}
        ], lr=config.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
        
        self.optimizer_d = optim.AdamW([
            {'params': self.msd.parameters()},
            {'params': self.mpd.parameters()},
            {'params': self.mrf.parameters()}
        ], lr=config.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
        
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
        try:
            # 배치 데이터 언패킹
            text_padded = batch['text_padded'].to(self.device)
            mel_padded = batch['mel_padded'].to(self.device)
            gate_padded = batch['gate_padded'].to(self.device)
            audio_padded = batch['audio_padded'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            
            # 디버깅을 위한 shape 출력
            print("\nInput shapes:")
            print(f"text_padded: {text_padded.shape}")
            print(f"mel_padded: {mel_padded.shape}")
            print(f"gate_padded: {gate_padded.shape}")
            print(f"audio_padded: {audio_padded.shape}")
            
            # Tacotron2 forward
            mel_outputs_postnet, mel_outputs, gate_outputs, _ = self.tacotron2(
                text_padded, text_lengths, mel_padded, mel_lengths)
            
            # Generator forward
            mel_for_generator = mel_outputs_postnet.transpose(1, 2)
            fake_audio = self.generator(mel_for_generator)
            
            # Discriminator forward
            msd_real_outputs, msd_real_features = self.msd(audio_padded)
            msd_fake_outputs, msd_fake_features = self.msd(fake_audio.detach())
            
            mpd_real_outputs, mpd_real_features = self.mpd(audio_padded)
            mpd_fake_outputs, mpd_fake_features = self.mpd(fake_audio.detach())
            
            # Loss 계산
            # Tacotron2 loss
            tacotron2_losses = self.criterion.tacotron2_loss(
                mel_outputs, mel_outputs_postnet, gate_outputs,
                mel_padded, gate_padded, mel_lengths
            )
            
            # HiFi-GAN loss
            real_outputs = msd_real_outputs + mpd_real_outputs
            fake_outputs = msd_fake_outputs + mpd_fake_outputs
            real_features = msd_real_features + mpd_real_features
            fake_features = msd_fake_features + mpd_fake_features
            
            hifigan_losses = self.criterion.hifi_gan_loss(
                audio_padded, fake_audio,
                real_outputs, fake_outputs,
                real_features, fake_features
            )
            
            # Optimizer step
            # Generator update
            self.optimizer_g.zero_grad()
            g_loss = tacotron2_losses['total_loss'] + hifigan_losses['generator_loss']
            g_loss.backward(retain_graph=True)
            self.optimizer_g.step()
            
            # Discriminator update
            self.optimizer_d.zero_grad()
            d_loss = hifigan_losses['discriminator_loss']
            d_loss.backward()
            self.optimizer_d.step()
            
            return {
                'total_loss': g_loss + d_loss,
                **tacotron2_losses,
                **hifigan_losses
            }
            
        except Exception as e:
            print(f"\nError in train_step: {str(e)}")
            print("\nBatch information:")
            print(f"Batch size: {len(batch)}")
            print(f"Batch contents: {[type(x) for x in batch]}")
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
        batch_size=4,
        learning_rate=0.0002,
        num_epochs=100,
        start_epoch=0,
        save_interval=10,
        checkpoint_dir='checkpoints',
        resume_checkpoint=None
    )
    
    trainer = Trainer(config)
    trainer.train() 