import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import torchaudio
from jamo import hangul_to_jamo
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
        sequence = []
        for char in text:
            if '가' <= char <= '힣':
                jamos = list(hangul_to_jamo(char))
                sequence.extend(jamos)
            else:
                sequence.append(char)
        return sequence
    
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
        text = item['original_script']
        sequence = self._text_to_sequence(text)
        sequence = [self.vocab.get(char, self.vocab['<unk>']) for char in sequence]
        sequence = torch.LongTensor(sequence)
        
        # 오디오 처리
        audio = torch.from_numpy(item['audio']['array']).float()
        original_sr = item['audio']['sampling_rate']
        
        if original_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr,
                new_freq=self.target_sr
            )
            audio = resampler(audio)
        
        # 멜 스펙트로그램 변환
        mel = self._wav_to_mel(audio.numpy())
        mel = torch.FloatTensor(mel).transpose(0, 1)
        
        return {
            'text': sequence,
            'text_length': sequence.size(0),
            'mel': mel,
            'mel_length': mel.size(0),
            'audio': audio
        }

    @staticmethod
    def collate_fn(batch):
        # 텍스트 패딩
        text_lengths = [x['text'].size(0) for x in batch]
        max_text_len = max(text_lengths)
        text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        for i, x in enumerate(batch):
            text = x['text']
            text_padded[i, :len(text)] = text
        
        # 멜 스펙트로그램 패딩
        mel_lengths = [x['mel'].size(0) for x in batch]
        max_mel_len = max(mel_lengths)
        mel_padded = torch.zeros(len(batch), 80, max_mel_len)
        for i, x in enumerate(batch):
            mel = x['mel']
            mel_padded[i, :, :mel.size(0)] = mel.transpose(0, 1)
        
        # 오디오 패딩
        audio_lengths = [x['audio'].size(0) for x in batch]
        max_audio_len = max(audio_lengths)
        audio_padded = torch.zeros(len(batch), 1, max_audio_len)
        for i, x in enumerate(batch):
            audio = x['audio']
            audio_padded[i, 0, :audio.size(0)] = audio
        
        # gate 패딩 생성
        gate_padded = torch.zeros(len(batch), max_mel_len)
        for i, length in enumerate(mel_lengths):
            gate_padded[i, length-1:] = 1
        
        # 길이 정보도 함께 반환
        text_lengths = torch.LongTensor(text_lengths)
        mel_lengths = torch.LongTensor(mel_lengths)
        
        return {
            'text_padded': text_padded,
            'mel_padded': mel_padded,
            'gate_padded': gate_padded,
            'audio_padded': audio_padded,
            'text_lengths': text_lengths,
            'mel_lengths': mel_lengths
        }