import hashlib
import torch
import torchaudio
import numpy as np
import os
from typing import Dict, Any, List
from joblib import Parallel, delayed
import librosa

DEFAULT_FEATURES_PARAMS = {
    "n_fft": 1024,
    "sample_rate": 16000,
    "n_mels": 128,
    "n_harmonics": 115,
}

class AudioEncoderMFCCHUTokenizer(object):
    
    def __init__(self, 
                 max_length: int = 256,
                 params: Dict[str, Any] = DEFAULT_FEATURES_PARAMS, 
                 cache_path: str = "./audio_features_cache"):
        
        self.max_length = max_length
        self.params = params
        self.cache_path = cache_path
        self.mean = None
        self.std = None
    
    def compute_speech_features(self, x):
        
        x = librosa.effects.preemphasis(x)

        mfcc = librosa.feature.mfcc(
            y=x, 
            n_fft=self.params["n_fft"], 
            sr=self.params["sample_rate"], 
            n_mfcc=13, 
            dct_type=2, 
            win_length=self.params["n_fft"], 
            hop_length=self.params["n_fft"] // 2, 
            norm='ortho', 
            lifter=22.0, 
            n_mels=self.params["n_mels"], 
            center=True, 
            fmin=20.0, 
            fmax=self.params["sample_rate"] / 2.0,
            window='hann',
        )

        f0, _, _ = librosa.pyin(
            y=x, 
            sr=self.params["sample_rate"], 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), 
            frame_length=self.params["n_fft"], 
            hop_length=self.params["n_fft"] // 2,
            win_length=self.params["n_fft"] // 4,
            center=True
        )

        S = np.abs(
            librosa.stft(
                x, 
                n_fft=self.params["n_fft"], 
                hop_length=self.params["n_fft"] // 2, 
                win_length=self.params["n_fft"], 
                window='hann', 
                center=True
            )
        )

        harmonics = np.arange(1, self.params["n_harmonics"] + 1)
        frequencies = librosa.fft_frequencies(sr=self.params["sample_rate"], n_fft=self.params["n_fft"])
        harmonic_energy = librosa.f0_harmonics(S, f0=f0, harmonics=harmonics, freqs=frequencies)

        with torch.no_grad():
            feats = torch.cat([torch.from_numpy(mfcc), torch.from_numpy(harmonic_energy)], dim=0)
            deltas = torchaudio.functional.compute_deltas(feats)
            ddeltas = torchaudio.functional.compute_deltas(deltas)

            feats_with_deltas = torch.cat([feats, deltas, ddeltas], dim=0)
            feats_with_deltas = feats_with_deltas.transpose(0, 1).contiguous()
            
            return feats_with_deltas
    
    def cache_dataset(self, paths: List[str], set_distribution: bool = True, n_jobs: int = 12):
        X = Parallel(n_jobs=n_jobs)(delayed(self.get_features)(x) for x in paths)
        lens = [x.shape[0] for x in X]

        X = torch.cat(X, dim=0)
        if set_distribution:
            self.mean = X.mean(dim=0)
            self.std = X.std(dim=0)
        return X, lens

    def get_features(self, path: str):
        hashed_name = hashlib.md5(path.encode('utf-8')).hexdigest()
        hashed_path = self.cache_path + '/' + f"{hashed_name}.bin"
        with torch.no_grad():
            if os.path.isfile(hashed_path):
                return torch.load(hashed_path, map_location='cpu')
            else:
                waveform, sample_rate = torchaudio.load(path, normalize=True, channels_first=True)
                waveform = waveform.float()
                    
                if len(waveform.shape) == 2:
                    waveform = torch.mean(waveform, dim=0).unsqueeze(dim=0)

                if sample_rate != self.params["sample_rate"]:
                    transform = torchaudio.transforms.Resample(sample_rate, self.params["sample_rate"])
                    waveform = transform(waveform)

                feats = self.compute_speech_features(waveform.numpy()[0])
                torch.save(feats, hashed_path)
                return feats
    
    def pad_features(self, x):
        l = len(x)
        att_mask = torch.ones((self.max_length, 1))

        if l > self.max_length:
            x = x[:self.max_length]
        elif l < self.max_length:
            mask_idx = torch.Tensor([i + l for i in range(self.max_length - l)]).long()
            att_mask = att_mask.index_fill_(0, mask_idx, 0.0)
            repeat = torch.zeros((self.max_length - l, x.shape[1]))
            x = torch.cat([x, repeat], dim=0)

        x = x.unsqueeze(dim=0)
        att_mask = att_mask.unsqueeze(dim=0).squeeze(dim=-1)
        return x, att_mask

    def tokenize(self, path: str):
        assert self.mean is not None and self.std is not None
        with torch.no_grad():
            x = (self.get_features(path) - self.mean) / (self.std + 1e-12)
            x, att_mask = self.pad_features(x)
            return x, att_mask
    
    def batch_tokenize(self, paths: str, n_jobs: int = 12):
        X = Parallel(n_jobs=n_jobs)(delayed(self.tokenize)(x) for x in paths)
        X_feats = [m for m, _ in X]
        att_masks = [a for _, a in X]
        X_feats = torch.cat(X_feats, dim=0)
        att_masks = torch.cat(att_masks, dim=0)
        return X_feats, att_masks