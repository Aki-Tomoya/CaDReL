import torch
from torch import nn
from torch.nn import functional as F
from common.models.transformer.attention import MultiHeadAttention, NormSelfAttention
from common.models.transformer.utils import GridPESine, PolarRPE, PositionWiseFeedForward, RelationalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, with_pe=None):
        super(EncoderLayer, self).__init__()

        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout,
                                        attention_module=NormSelfAttention,
                                        attention_module_kwargs={'with_pe': with_pe})

        self.cross_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        
        self.self_pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.cross_pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, gird_features, object_queries, geometric_attention=None):

        grid_att = self.self_att(gird_features, gird_features, gird_features, attention_weights=geometric_attention)

        object_att = self.cross_att(object_queries, gird_features, gird_features)

        gird_ff = self.self_pwff(grid_att)
        object_ff = self.cross_pwff(object_att)
        return gird_ff, object_ff

class TransformerEncoder(nn.Module):
    def __init__(self, N, prefix_length=10, with_pe=None, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.with_pe = with_pe

        self.grid_proj = nn.Sequential(
            nn.Linear(2560, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

        if with_pe and with_pe == 'rpe':
            self.rddpe = PolarRPE(k=3, h=h, d_k=d_k, d_r=256, device=device)

        self.object_queries = nn.Parameter(torch.randn(prefix_length, d_model), requires_grad=True)

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, with_pe) for _ in range(N)])

    def forward(self, grid_features):
        # input (b_s, seq_len)
        grid_features = self.grid_proj(grid_features)

        b_s = grid_features.shape[0]
        query_shape = self.object_queries.shape
        object_queries = self.object_queries.unsqueeze(0).expand(b_s, *query_shape).contiguous()

        geometric_attention = None
        if self.with_pe and self.with_pe == 'rpe':
            geometric_attention = self.rddpe(b_s)

        for l in self.layers:
            grid_features, object_queries = l(grid_features, object_queries, geometric_attention)

        return grid_features, object_queries

def build_encoder(N, prefix_length=10, with_pe=None, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
    Encoder = TransformerEncoder(N, prefix_length, with_pe, device, d_model, d_k, d_v, h, d_ff, dropout)
    
    return Encoder