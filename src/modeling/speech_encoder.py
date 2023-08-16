import torch
from torch import nn
import math
from vector_quantize_pytorch import ResidualVQ, GroupedResidualVQ
from torch.nn import functional as F

# Modified for compatibility with TransformerEncoder evaluation forward pass
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias_enabled=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias_enabled = bias_enabled

        self.weight = nn.Parameter(torch.ones(d))
        self.register_parameter("weight", self.weight)

        #if self.bias:
        self.bias = nn.Parameter(torch.zeros(d))
        self.register_parameter("bias", self.bias)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias_enabled:
            return self.weight * x_normed + self.bias

        return self.weight * x_normed
    
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
                 raw_features_size_1=39,
                 raw_features_size_2=120,
                 nheads=8, 
                 dropout=0.2,
                 pos_enc_drop=0.1,
                 codebook_dim=32,
                 num_quantizer=2,
                 threshold_ema_dead_code=2):
        super(AudioEncoderMFCCHU, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        self.vq_mfcc = GroupedResidualVQ(
            dim = raw_features_size_1,
            codebook_size = vocab_size,
            codebook_dim = codebook_dim,
            num_quantizers = num_quantizer,
            threshold_ema_dead_code = threshold_ema_dead_code,
            kmeans_init = True,   # set to True
            kmeans_iters = 10,
            groups=3,
        )

        self.vq_harm = GroupedResidualVQ(
            dim = raw_features_size_2,
            codebook_size = vocab_size,
            codebook_dim = codebook_dim,
            num_quantizers = num_quantizer,
            threshold_ema_dead_code = threshold_ema_dead_code,
            kmeans_init = True,   # set to True
            kmeans_iters = 10,
            groups=3,
        )

        self.emb_dim = emb_dim
        self.dropout = dropout
        self.raw_features_size_1 = raw_features_size_1
        self.raw_features_size_2 = raw_features_size_2
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=pos_enc_drop)
        self.projection_out_1 = nn.Sequential(nn.Linear(raw_features_size_1, emb_dim), nn.GELU(), nn.Dropout(p=self.dropout), nn.Linear(emb_dim, emb_dim))
        self.projection_out_2 = nn.Sequential(nn.Linear(raw_features_size_2, emb_dim), nn.GELU(), nn.Dropout(p=self.dropout), nn.Linear(emb_dim, emb_dim))

        self.norm_out_1 = RMSNorm(emb_dim)
        self.norm_out_2 = RMSNorm(emb_dim)

        self.transf_layer = nn.TransformerEncoderLayer(d_model=emb_dim, dim_feedforward=emb_dim*2, nhead=nheads, batch_first=True, norm_first=True, dropout=self.dropout, activation=F.gelu)
        self.transf_layer.norm1 = RMSNorm(emb_dim)
        self.transf_layer.norm2 = RMSNorm(emb_dim)
        self.transf_enc = nn.TransformerEncoder(self.transf_layer, num_layers=n_layers, norm=RMSNorm(emb_dim))

    def forward(self, features, attn_masks):
        mfcc = features[:, :, :self.raw_features_size_1]
        harm = features[:, :, self.raw_features_size_1:]

        x_mfcc, _, vq_loss_1 = self.vq_mfcc(mfcc)
        vq_loss_1 = vq_loss_1.mean()
        x_mfcc = self.projection_out_1(x_mfcc)
        x_mfcc = x_mfcc + self.norm_out_1(x_mfcc)
        x_mfcc = self.pos_encoder(x_mfcc)

        x_harm, _, vq_loss_2 = self.vq_harm(harm)
        vq_loss_2 = vq_loss_2.mean()
        x_harm = self.projection_out_2(x_harm)
        x_harm = x_harm + self.norm_out_2(x_harm)
        x_harm = self.pos_encoder(x_harm)

        x = (x_harm + x_mfcc) / 2.0
        vq_loss = (vq_loss_1 + vq_loss_2) / 2.0

        x = self.transf_enc(x, src_key_padding_mask=attn_masks)
        
        input_mask_expanded = attn_masks.unsqueeze(-1).expand(x.size()).float()
        x = torch.sum(x * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return x, vq_loss