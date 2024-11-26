import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
import traceback

from dataset import KSSTTSDataset
from tacotron2 import Tacotron2
from hifi_gan import Generator, MultiScaleDiscriminator, MultiPeriodDiscriminator, MRFDiscriminator
from loss import TTSLoss

class Trainer:
    def __init__(self, config):
        # 기본 설정
        self.config = config
        self.start_epoch = config.start_epoch
        self.epochs = config.num_epochs
        self.batch_size = config.batch_size

        # CUDA 메모리 설정
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = True
            
            # 사용할 GPU 인덱스 리스트
            self.gpu_ids = [int(i) for i in config.device_num.split(',')]
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")  # 주 GPU 설정
            print(f"Using GPUs: {self.gpu_ids}")
        else:
            self.device = torch.device("cpu")
            self.gpu_ids = []
        print(f"Primary device: {self.device}")

        # Dataset 초기화
        self.train_dataset = KSSTTSDataset(split='train')
        self.valid_dataset = KSSTTSDataset(split='valid')
        
        # DataLoader 설정
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=KSSTTSDataset.collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=KSSTTSDataset.collate_fn,
            num_workers=2,
            pin_memory=True
        )

        # mel_channels 저장
        self.mel_channels = config.mel_channels
        print(f"Initializing models with mel_channels={self.mel_channels}")
        
        # 모델 초기화 전에 mel_channels 확인
        assert self.mel_channels == 80, f"mel_channels must be 80, but got {self.mel_channels}"
        
        # 모델 초기화
        self.tacotron2 = Tacotron2(
            n_mel_channels=self.mel_channels,  # config.mel_channels 대신 self.mel_channels 사용
            vocab_size=len(self.train_dataset.vocab),
            embedding_dim=256,
            encoder_n_convolutions=3,
            encoder_kernel_size=5,
            attention_rnn_dim=512,
            attention_dim=128
        ).to(self.device)
        
        # Generator 초기화 시 동일한 mel_channels 사용
        self.generator = Generator(input_size=self.mel_channels).to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.mrf = MRFDiscriminator().to(self.device)

        # Gradient Checkpointing 활성화 (DataParallel 전에)
        if hasattr(self.tacotron2, 'encoder') and hasattr(self.tacotron2.encoder, 'lstm'):
            self.tacotron2.encoder.lstm.requires_grad_(True)
        if hasattr(self.tacotron2, 'decoder') and hasattr(self.tacotron2.decoder, 'decoder_rnn'):
            self.tacotron2.decoder.decoder_rnn.requires_grad_(True)

        # Multi-GPU 설정 (GPU가 여러 개인 경우)
        if len(self.gpu_ids) > 1:
            self.tacotron2 = nn.DataParallel(self.tacotron2, device_ids=self.gpu_ids)
            self.generator = nn.DataParallel(self.generator, device_ids=self.gpu_ids)
            self.msd = nn.DataParallel(self.msd, device_ids=self.gpu_ids)
            self.mpd = nn.DataParallel(self.mpd, device_ids=self.gpu_ids)
            self.mrf = nn.DataParallel(self.mrf, device_ids=self.gpu_ids)

        # Loss criterion 초기화
        self.criterion = TTSLoss(device=self.device)
        
        # Optimizer 초기화
        self.optimizer_t2 = optim.Adam(self.tacotron2.parameters(), 
                                      lr=config.learning_rate, 
                                      weight_decay=1e-6)
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

        # 체크포인트 디렉토리 설정
        self.checkpoint_dir = Path(__file__).parent / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision scaler 초기화
        self.scaler = torch.cuda.amp.GradScaler()

    def save_checkpoint(self, epoch, loss):
        # DataParallel 모델에서 단일 GPU용 state_dict 추출
        def get_state_dict(model):
            if isinstance(model, nn.DataParallel):
                return model.module.state_dict()
            return model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'tacotron2_state_dict': get_state_dict(self.tacotron2),
            'generator_state_dict': get_state_dict(self.generator),
            'msd_state_dict': get_state_dict(self.msd),
            'mpd_state_dict': get_state_dict(self.mpd),
            'mrf_state_dict': get_state_dict(self.mrf),
            'optimizer_t2_state_dict': self.optimizer_t2.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'loss': loss,
            'config': {
                'n_mel_channels': 80,
                'vocab_size': len(self.train_dataset.vocab),
                'embedding_dim': 256,
                'encoder_n_convolutions': 3,
                'encoder_kernel_size': 5,
                'attention_rnn_dim': 512,
                'attention_dim': 128
            }
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 모델 로드 (DataParallel 여부와 관계없이)
        if isinstance(self.tacotron2, nn.DataParallel):
            self.tacotron2.module.load_state_dict(checkpoint['tacotron2_state_dict'])
            self.generator.module.load_state_dict(checkpoint['generator_state_dict'])
            self.msd.module.load_state_dict(checkpoint['msd_state_dict'])
            self.mpd.module.load_state_dict(checkpoint['mpd_state_dict'])
            self.mrf.module.load_state_dict(checkpoint['mrf_state_dict'])
        else:
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
            torch.cuda.empty_cache()
            
            text_padded, input_lengths, mel_padded, gate_padded, mel_lengths, audio_padded = [
                x.to(self.device) for x in batch
            ]
            
            # Gradient Accumulation을 위한 스케일링
            accumulation_steps = self.config.gradient_accumulation_steps
            batch_size = text_padded.size(0)
            sub_batch_size = max(1, batch_size // accumulation_steps)
            
            total_loss = 0
            tacotron_losses = None
            hifi_losses = None
            
            for i in range(accumulation_steps):
                start_idx = i * sub_batch_size
                end_idx = min((i + 1) * sub_batch_size, batch_size)
                
                if start_idx >= end_idx:
                    continue
                    
                with torch.cuda.amp.autocast():
                    # Tacotron2 forward pass
                    mel_output, mel_output_postnet, gate_output, alignments = self.tacotron2(
                        text_padded[start_idx:end_idx],
                        input_lengths[start_idx:end_idx],
                        mel_padded[start_idx:end_idx],
                        mel_lengths[start_idx:end_idx]
                    )
                    
                    # Adjust lengths to match
                    max_len = mel_padded[start_idx:end_idx].size(2)
                    if mel_output.size(1) > max_len:
                        mel_output = mel_output[:, :max_len, :]
                        mel_output_postnet = mel_output_postnet[:, :max_len, :]
                        gate_output = gate_output[:, :max_len]
                    elif mel_output.size(1) < max_len:
                        pad_len = max_len - mel_output.size(1)
                        mel_output = F.pad(mel_output, (0, 0, 0, pad_len))
                        mel_output_postnet = F.pad(mel_output_postnet, (0, 0, 0, pad_len))
                        gate_output = F.pad(gate_output, (0, pad_len))
                    
                    # Loss 계산
                    tacotron_losses = self.criterion.tacotron2_loss(
                        mel_output, mel_output_postnet, gate_output,
                        mel_padded[start_idx:end_idx],
                        gate_padded[start_idx:end_idx],
                        mel_lengths[start_idx:end_idx]
                    )
                    
                    # Generator forward pass with adjusted mel
                    mel_for_generator = mel_output_postnet.detach()
                    
                    # mel spectrogram 차원 조정 (B, T, C) -> (B, C, T)
                    mel_for_generator = mel_for_generator.transpose(1, 2)
                    
                    # HiFi-GAN forward pass
                    fake_audio = self.generator(mel_for_generator)
                    
                    # Adjust audio length and channels
                    target_audio = audio_padded[start_idx:end_idx]
                    
                    # 오디오 길이 조정
                    if fake_audio.size(-1) > target_audio.size(-1):
                        fake_audio = fake_audio[..., :target_audio.size(-1)]
                    elif fake_audio.size(-1) < target_audio.size(-1):
                        fake_audio = F.pad(fake_audio, (0, target_audio.size(-1) - fake_audio.size(-1)))
                    
                    # 채널 수 조정 (mono로 변환)
                    if target_audio.size(1) > 1:
                        target_audio = target_audio.mean(dim=1, keepdim=True)
                    if fake_audio.size(1) > 1:
                        fake_audio = fake_audio.mean(dim=1, keepdim=True)
                    
                    # Print shapes for debugging
                    print(f"\nDebug shapes after adjustment:")
                    print(f"mel_for_generator shape: {mel_for_generator.shape}")
                    print(f"fake_audio shape: {fake_audio.shape}")
                    print(f"target_audio shape: {target_audio.shape}")
                    
                    # Discriminator forward passes
                    try:
                        msd_real, msd_fake, fmap_msd_real, fmap_msd_fake = self.msd(target_audio, fake_audio.detach())
                        mpd_real, mpd_fake, fmap_mpd_real, fmap_mpd_fake = self.mpd(target_audio, fake_audio.detach())
                        mrf_real, mrf_fake, fmap_mrf_real, fmap_mrf_fake = self.mrf(target_audio, fake_audio.detach())
                    except RuntimeError as e:
                        print(f"Discriminator error: {str(e)}")
                        print(f"Target audio shape: {target_audio.shape}")
                        print(f"Fake audio shape: {fake_audio.shape}")
                        raise e
                    
                    # HiFi-GAN losses
                    hifi_losses = self.criterion.hifi_gan_loss(
                        target_audio,
                        fake_audio,
                        [msd_real, mpd_real, mrf_real],
                        [msd_fake, mpd_fake, mrf_fake],
                        [fmap_msd_real, fmap_mpd_real, fmap_mrf_real],
                        [fmap_msd_fake, fmap_mpd_fake, fmap_mrf_fake]
                    )
                    
                    # Scale losses
                    loss = (tacotron_losses['total_loss'] + 
                           hifi_losses['generator_loss'] + 
                           hifi_losses['discriminator_loss']) / accumulation_steps
                    
                # Backward pass with gradient accumulation
                self.scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0 or (i + 1) == batch_size:
                    # Update weights
                    self.scaler.step(self.optimizer_t2)
                    self.scaler.step(self.optimizer_g)
                    self.scaler.step(self.optimizer_d)
                    self.scaler.update()
                    
                    # Zero gradients
                    self.optimizer_t2.zero_grad()
                    self.optimizer_g.zero_grad()
                    self.optimizer_d.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                
                # 중간 결과물 메모리 해제
                del mel_output, mel_output_postnet, gate_output, fake_audio
                del msd_real, msd_fake, mpd_real, mpd_fake, mrf_real, mrf_fake
                torch.cuda.empty_cache()
            
            if tacotron_losses is None:
                raise ValueError("No valid sub-batches were processed")
            
            return {
                'total_loss': float(total_loss),
                'mel_loss': float(tacotron_losses['mel_loss']),
                'gate_loss': float(tacotron_losses['gate_loss']),
                'generator_loss': float(hifi_losses['generator_loss']),
                'discriminator_loss': float(hifi_losses['discriminator_loss'])
            }

        except Exception as e:
            print(f"Error in train_step: {str(e)}")
            traceback.print_exc()
            raise e

    def train(self):
        avg_loss = float('inf')
        epoch = self.start_epoch
        
        try:
            for epoch in range(self.start_epoch, self.epochs):
                # 모델들을 학습 모드로 설정
                self.tacotron2.train()
                self.generator.train()
                self.msd.train()
                self.mpd.train()
                self.mrf.train()
                
                total_loss = 0
                batch_count = 0
                
                # tqdm 설정 개선
                progress_bar = tqdm(
                    self.train_loader,
                    desc=f'Epoch {epoch + 1}/{self.epochs}',
                    ncols=100,
                    position=0,
                    leave=True
                )
                
                for batch in progress_bar:
                    try:
                        losses = self.train_step(batch)
                        total_loss += losses['total_loss']
                        batch_count += 1
                        
                        # 현재까지의 평균 손실 계산
                        current_avg_loss = total_loss / batch_count
                        
                        # tqdm 진행바 업데이트
                        progress_bar.set_postfix({
                            'loss': f"{losses['total_loss']:.4f}",
                            'mel': f"{losses['mel_loss']:.4f}",
                            'gate': f"{losses['gate_loss']:.4f}",
                            'gen': f"{losses['generator_loss']:.4f}",
                            'disc': f"{losses['discriminator_loss']:.4f}"
                        })
                        
                    except Exception as e:
                        print(f"\n예기치 않은 오류가 발생했습니다: {str(e)}")
                        traceback.print_exc()
                        continue
                
                # 배치가 하나라도 성공적으로 처리되었다면
                if batch_count > 0:
                    avg_loss = total_loss / batch_count
                    print(f'\nEpoch {epoch + 1} 평균 손실: {avg_loss:.4f}')
                    
                    # 체크포인트 저장
                    self.save_checkpoint(epoch + 1, avg_loss)
                else:
                    print(f'\nEpoch {epoch + 1}에서 성공적으로 처리된 배치가 없습니다.')
            
        except KeyboardInterrupt:
            print('\n학습이 사용자에 의 중단되었습니다.')
        except Exception as e:
            print(f'\n예기치 않은 오류가 발생했습니다: {str(e)}')
            traceback.print_exc()
        finally:
            # 마지막 상태 저장
            print(f'\n최종 체크포인트 저장 중... (Epoch: {epoch + 1}, Loss: {avg_loss:.4f})')
            self.save_checkpoint(epoch + 1, avg_loss)
            print('학습 종료')

if __name__ == '__main__':
    from argparse import Namespace
    
    mel_channels = 80  # 모든 모델에서 사용할 일관된 mel channels 수
    
    config = Namespace(
        device_num='3,4',
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=5,
        start_epoch=0,
        save_interval=1,
        checkpoint_dir='checkpoints',
        resume_checkpoint=None,
        gradient_accumulation_steps=4,
        max_audio_length=5,
        target_sr=16000,
        mel_channels=80,  # 명시적으로 설정
    )
    
    trainer = Trainer(config)
    trainer.train() 