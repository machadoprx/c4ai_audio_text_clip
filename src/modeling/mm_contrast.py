import torch
from torch import nn
import torch.nn.functional as F

class AudioTextContrastive(nn.Module):

    def __init__(self, 
                 text_encoder, 
                 audio_encoder, 
                 in_features_text=384, 
                 in_features_audio=16, 
                 wide_proj=1024, 
                 proj_size=128,
                 hidden_size=384,
                 rate=0.1,):
        super(AudioTextContrastive, self).__init__()
        
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder

        self.mods_proj = nn.Sequential(
            nn.Dropout(p=rate), 
            nn.Linear(hidden_size, hidden_size), 
            nn.GELU(), 
            nn.Linear(hidden_size, wide_proj)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(in_features_text, hidden_size),  
            nn.GELU(), 
            nn.Linear(hidden_size, hidden_size),
            self.mods_proj,
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(in_features_audio, hidden_size), 
            nn.GELU(), 
            nn.Linear(hidden_size, hidden_size),
            self.mods_proj,
        )

        self.linear = nn.Linear(wide_proj, proj_size, bias=False)
        
        self.alpha = nn.Sequential(
            nn.Dropout(p=rate),
            nn.Linear(2 * wide_proj, 2)
        )
        self.wide_proj = wide_proj
        self.rate = rate
        
    def forward(self, inp):

        sentences, audio_input, multimodal = inp
        
        assert sentences != None or audio_input != None or multimodal != None
        self.audio_encoder.eval()
        self.text_encoder.eval()

        x_text = None
        x_text_wide = None
        if sentences != None:
            with torch.no_grad():
                x_text_wide = self.text_encoder(sentences)
            x_text_wide = F.normalize(self.text_proj(x_text_wide), dim=-1)
            x_text = F.normalize(self.linear(x_text_wide), dim=-1)
            
        x_audio = None
        x_audio_wide = None
        vq_loss = None
        if audio_input != None:
            with torch.no_grad():
                x_audio_wide, vq_loss = self.audio_encoder(**audio_input)
            x_audio_wide = F.normalize(self.audio_proj(x_audio_wide), dim=-1)
            x_audio = F.normalize(self.linear(x_audio_wide), dim=-1)
        
        x_mult_text = None
        x_mult_text_wide = None
        x_mult_audio = None
        x_mult_audio_wide = None
        
        # Approximate text and audio, and make sum of vectors point to correct cls
        if multimodal != None:
            with torch.no_grad():
                x_mult_text = self.text_encoder(multimodal['sentences'])
            x_mult_text = self.text_proj(x_mult_text)
            with torch.no_grad():
                x_mult_audio, _ = self.audio_encoder(**multimodal['audio_input'])
            x_mult_audio = self.audio_proj(x_mult_audio)
            
            alphas = self.alpha(torch.cat([x_mult_text, x_mult_audio], dim=-1))
            alphas = F.softmax(alphas, dim=-1).unsqueeze(dim=1)

            # View 1
            x_mult_text_wide = alphas[:, :, 0] * F.normalize(x_mult_text, dim=-1)
            x_mult_text = alphas[:, :, 0] * F.normalize(self.linear(x_mult_text), dim=-1)
            
            # View 2
            x_mult_audio_wide = alphas[:, :, 1] * F.normalize(x_mult_audio, dim=-1)
            x_mult_audio = alphas[:, :, 1] * F.normalize(self.linear(x_mult_audio), dim=-1)

        return {
            "x_text": x_text,
            "x_text_wide": x_text_wide,
            "x_audio": x_audio,
            "x_audio_wide": x_audio_wide,
            "x_mult_text": x_mult_text,
            "x_mult_text_wide": x_mult_text_wide,
            "x_mult_audio": x_mult_audio,
            "x_mult_audio_wide": x_mult_audio_wide,
            "vq_loss": vq_loss,
        }