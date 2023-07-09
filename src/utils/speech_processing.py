import hashlib
import torch
import torchaudio
import numpy as np
import os
from typing import Dict, Any
from joblib import Parallel, delayed

sample_rate = 16000
n_fft = 400.0
frame_length = n_fft / sample_rate * 1000.0
frame_shift = frame_length / 2.0

DEFAULT_MFCC_PARAMS = {
    "channel": 0,
    "dither": 0.0,
    "window_type": "hanning",
    "frame_length": frame_length,
    "frame_shift": frame_shift,
    "remove_dc_offset": False,
    "round_to_power_of_two": False,
    "sample_frequency": sample_rate,
}

DEFAULT_PITCH_PARAMS = {
    "frame_length": frame_length,
    "frame_shift": frame_shift,
    "sample_rate": sample_rate,
}

class AudioEncoderMFCCHUTokenizer(object):
    
    def __init__(self, 
                 max_length: int = 128,
                 max_pool_window_size: int = 4,
                 mfcc_params: Dict[str, Any] = DEFAULT_MFCC_PARAMS, 
                 pitch_params: Dict[str, int] = DEFAULT_PITCH_PARAMS,
                 cache_path: str = "./preprocessed_audio_cache_new"):
        
        self.max_length = max_length
        self.mfcc_params = mfcc_params
        self.pitch_params = pitch_params
        self.cache_path = cache_path
        self.max_pool_window_size = max_pool_window_size
        self.mean = None
        self.std = None
    
    def _get_mfcc_feats(self, x):
        pitch = torchaudio.functional.compute_kaldi_pitch(x, **self.pitch_params).squeeze(dim=0)
        
        x = x.view(1, -1)

        mfccs = torchaudio.compliance.kaldi.mfcc(
            x,
            **self.mfcc_params
        )  # (time, freq)
        
        try:
            mfccs = torch.cat([mfccs, pitch], dim=-1)
        except:
            mfccs = torch.cat([mfccs, torch.Tensor(np.zeros((mfccs.shape[0], 2)))], dim=-1)
        
        mfccs_z = torch.Tensor(np.zeros(((mfccs.shape[0] // self.max_pool_window_size) + 1, mfccs.shape[1])))
        
        for i in range(len(mfccs) // self.max_pool_window_size): # Max pooling over time to reduce sequence size
            mfcc_win = mfccs[i * self.max_pool_window_size:(i + 1) * self.max_pool_window_size]
            norms = [np.linalg.norm(v[:-2]) for v in mfcc_win]
            argmax = np.argmax(np.array(norms))
            mfccs_z[i] = mfcc_win[argmax]
                
        mfccs = mfccs_z.transpose(0, 1)  # (freq, time)
        deltas = torchaudio.functional.compute_deltas(mfccs)
        ddeltas = torchaudio.functional.compute_deltas(deltas)
        concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
        concat = concat.transpose(0, 1).contiguous()
        
        return concat
    
    def cache_dataset(self, paths: str, set_distribution: bool = True, n_jobs: int = 12):
        X = Parallel(n_jobs=n_jobs)(delayed(self.mfcc_feature_loader)(x) for x in paths)
        X = torch.cat(X, dim=0)
        if set_distribution:
            self.mean = X.mean(dim=0).cuda()
            self.std = X.std(dim=0).cuda()

    def mfcc_feature_loader(self,
                            path: str):
        hashed_name = hashlib.md5(path.encode('utf-8')).hexdigest()
        hashed_path = self.cache_path + '/' + f"{hashed_name}.bin"
        with torch.no_grad():
            if os.path.isfile(hashed_path):
                return torch.load(hashed_path)
            else:
                waveform, sample_rate = torchaudio.load(path, normalize=True, channels_first=True)
                waveform = waveform.float()
                    
                if len(waveform.shape) == 2:
                    waveform = torch.mean(waveform, dim=0).unsqueeze(dim=0)

                if sample_rate != self.mfcc_params["sample_frequency"]:
                    transform = torchaudio.transforms.Resample(sample_rate, self.mfcc_params["sample_frequency"])
                    waveform = transform(waveform)

                mfcc = self._get_mfcc_feats(waveform, self.max_pool_window_size, self.mfcc_params, self.pitch_params)
                torch.save(mfcc, hashed_path)
                return mfcc
    
    def tokenize(self, path: str):
        assert self.mean != None and self.std != None

        with torch.no_grad():
            mfcc = (self.mfcc_feature_loader(path) - self.mean) / (self.std + 1e-10)
        
            l = len(mfcc)
            att_mask = torch.ones((self.max_length, 1))

            if l > self.max_length:
                mfcc = mfcc[:self.max_length]
            elif l < self.max_length:
                mask_idx = torch.Tensor([i + l for i in range(self.max_length - l)]).long()
                att_mask = att_mask.index_fill_(0, mask_idx, 0.0)
                repeat = torch.zeros((self.max_length - l, mfcc.shape[1]))
                mfcc = torch.cat([mfcc, repeat], dim=0)

            mfcc = mfcc.unsqueeze(dim=0)
            att_mask = att_mask.unsqueeze(dim=0).squeeze(dim=-1)
            return mfcc, att_mask
    
    def batch_tokenize(self, paths: str, n_jobs: int = 12):
        X = Parallel(n_jobs=n_jobs)(delayed(self.tokenize)(x) for x in paths)
        mfccs = [m for m, _ in X]
        att_masks = [a for _, a in X]
        mfccs = torch.cat(mfccs, dim=0)
        att_masks = torch.cat(att_masks, dim=0)
        return mfccs, att_masks