import torch
from torch import nn
import math
from vector_quantize_pytorch import ResidualVQ

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AudioEncoderMFCCHU(nn.Module):

    def __init__(self, 
                 vocab_size,
                 emb_dim=768, 
                 n_layers=6, 
                 max_length=800,
                 raw_features_size=45,
                 nheads=8, 
                 dropout=0.2,
                 pos_enc_drop=0.1,
                 codebook_dim=128,
                 num_quantizer=4,
                 threshold_ema_dead_code=2):
        super(AudioEncoderMFCCHU, self).__init__()
        
        self.vocab_size = vocab_size
        
        self.max_length = max_length
        
        self.vq = ResidualVQ(
            dim = raw_features_size,
            codebook_size = self.vocab_size,
            codebook_dim = codebook_dim,
            num_quantizers = num_quantizer,
            threshold_ema_dead_code = threshold_ema_dead_code,
        )
            
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=pos_enc_drop)
        self.project = nn.Sequential(nn.Linear(raw_features_size, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.transf_layer = nn.TransformerEncoderLayer(d_model=emb_dim, dim_feedforward=emb_dim*4, nhead=nheads, batch_first=True, norm_first=True, dropout=self.dropout)
        self.transf_enc = nn.TransformerEncoder(self.transf_layer, num_layers=n_layers, norm=nn.LayerNorm(emb_dim))
        
        self.norm_feats = nn.LayerNorm(raw_features_size)

    def forward(self, features, attn_masks):
        
        features = self.norm_feats(features)
        qtz_feats, _, vq_loss = self.vq(features)
        vq_loss = vq_loss.mean()
            
        seq_lens = 1 / torch.sum(attn_masks, dim=-1)
        seq_lens = seq_lens.unsqueeze(dim=-1)

        x = self.project(qtz_feats)
        x = self.pos_encoder(x)

        x = self.transf_enc(x, src_key_padding_mask=attn_masks)
        x = seq_lens * torch.sum(x, dim=1)
        
        return x, vq_loss