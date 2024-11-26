import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import torchaudio
import librosa

class KSSTTSDataset(Dataset):
    def __init__(self, split='train', valid_size=0.1, seed=42, target_sr=22050):
        super().__init__()
        
        # 기본 설정
        self.target_sr = target_sr
        self.vocab = None
        
        # 데이터셋 로드
        dataset = load_dataset("Bingsu/KSS_Dataset")
        full_dataset = dataset['train']
        
        # train/valid 분할
        train_idx, valid_idx = train_test_split(
            range(len(full_dataset)),
            test_size=valid_size,
            random_state=seed
        )
        
        self.indices = train_idx if split == 'train' else valid_idx
        self.dataset = full_dataset
        
        # vocab 초기화
        self._initialize_vocab()
        
        # Mel spectrogram 변환을 위한 transform 초기화
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0,
            f_max=8000,
            power=1
        )
        
        # log mel spectrogram을 위한 처리
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
    def _initialize_vocab(self):
        all_texts = [self.dataset[idx]['original_script'] for idx in self.indices]
        unique_tokens = set()
        
        for text in all_texts:
            sequence = self._text_to_sequence(text)
            unique_tokens.update(sequence)
            
        self.vocab = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        self.vocab['<pad>'] = len(self.vocab)
        self.vocab['<unk>'] = len(self.vocab)
    
    def _text_to_sequence(self, text):
        # 자모 분해 없이 텍스트를 그대로 시퀀스로 변환
        return list(text)
    
    def _wav_to_mel(self, wav):
        stft = librosa.stft(wav, n_fft=1024, hop_length=256, win_length=1024)
        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(stft)**2,
            sr=self.target_sr,
            n_mels=80,
            fmin=0,
            fmax=8000
        )
        mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
        mel_spec = np.clip(mel_spec, a_min=-4, a_max=4)
        mel_spec = (mel_spec + 4) / 8
        return mel_spec

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]
        
        # 텍스트 처리
        text = item['decomposed_script']
        sequence = self._text_to_sequence(text)
        sequence = [self.vocab.get(char, self.vocab['<unk>']) for char in sequence]
        sequence = torch.LongTensor(sequence)
        
        # 오디오 처리
        audio = torch.FloatTensor(item['audio']['array']).unsqueeze(0)  # [1, T]
        
        # 오디오 길이 제한 (5초로 축소)
        max_audio_length = 7 * self.target_sr
        if audio.size(-1) > max_audio_length:
            start = torch.randint(0, audio.size(-1) - max_audio_length, (1,))
            audio = audio[:, start:start + max_audio_length]
        
        # Mel spectrogram 계산 전 오디오 다운샘플링
        if self.target_sr > 16000:
            audio = torchaudio.transforms.Resample(
                self.target_sr, 16000
            )(audio)
            self.target_sr = 16000
        
        # 멜스펙트로그램 계산
        mel = self.mel_transform(audio)  # [n_mels, T]
        mel = self.amplitude_to_db(mel)
        
        # 정규화
        mel = (mel + 80) / 80
        
        # print(f"Mel shape in __getitem__: {mel.shape}")
        
        return {
            'text': sequence,
            'mel': mel,  # [n_mels, T] 형태로 반환
            'audio': audio
        }

    @staticmethod
    def collate_fn(batch):
        # 텍스트 처리
        input_lengths = [len(x['text']) for x in batch]
        max_input_len = max(input_lengths)
        
        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        for i, x in enumerate(batch):
            text = x['text']
            text_padded[i, :len(text)] = text

        # 멜스펙트로그램 처리
        mel_lengths = [x['mel'].size(-1) for x in batch]
        max_mel_len = max(mel_lengths)
        
        # mel_padded 차원 수정 [batch_size, n_mels, time]
        mel_padded = torch.zeros(len(batch), 80, max_mel_len)
        gate_padded = torch.zeros(len(batch), max_mel_len)
        
        for i, x in enumerate(batch):
            mel = x['mel']
            # mel의 shape 출력
            # print(f"Mel shape before processing: {mel.shape}")
            
            # mel의 shape를 [batch_size, n_mels, time]으로 변경
            if mel.dim() == 3:  # [1, 80, T]
                mel = mel.squeeze(0)  # [80, T]
            elif mel.dim() == 2 and mel.size(0) != 80:  # [T, 80]
                mel = mel.transpose(0, 1)  # [80, T]
                
            # print(f"Mel shape after processing: {mel.shape}")
            
            # 패딩
            cur_len = mel.size(1)
            mel_padded[i, :, :cur_len] = mel
            gate_padded[i, cur_len-1:] = 1

        # 오디오 처리
        audio_lengths = [x['audio'].size(-1) for x in batch]
        max_audio_len = max(audio_lengths)
        
        audio_padded = torch.zeros(len(batch), 1, max_audio_len)
        for i, x in enumerate(batch):
            audio = x['audio']
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            audio_padded[i, :, :audio.size(-1)] = audio

        return (text_padded, torch.LongTensor(input_lengths),
                mel_padded, gate_padded, torch.LongTensor(mel_lengths),
                audio_padded)