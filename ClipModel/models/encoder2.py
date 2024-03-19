import torch
from torch import nn
from torch.nn import functional as F
from common.models.transformer.attention import ChannelAttention, NormSelfAttention, MultiHeadAttention, AdapterAttention, RPEAttention
from common.models.transformer.utils import PolarRPE, DWConv, GridPESine, PositionWiseFeedForward, RelationalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, with_pe=None, mid_dim=40, local=False):
        super(EncoderLayer, self).__init__()

        # self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout,
        #                                 attention_module=AdapterAttention,
        #                                 attention_module_kwargs={'mid_dim': mid_dim, 'with_pe': with_pe})

        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout,
                                        attention_module=NormSelfAttention,
                                        attention_module_kwargs={'with_pe': with_pe})

        # self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout,
        #                                 attention_module=RPEAttention)
        
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, local=local)

    def forward(self, queries, keys, values, attention_mask=None, geometric_attention=None, grid_pos=None):
        if grid_pos:
            queries = queries + grid_pos
            keys = keys + grid_pos
        
        spatial_att = self.mhatt(queries, keys, values, attention_mask, attention_weights=geometric_attention)
        out = self.pwff(spatial_att)

        return out

class TransformerEncoder(nn.Module):
    def __init__(self, N, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, 
                 with_pe=None, mid_dim=40, multi_level = False, local=False, device='cuda'):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.multi_level = multi_level
        self.with_pe = with_pe

        self.in_proj_model = nn.Sequential(
            nn.Linear(d_in, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                         with_pe, mid_dim, local) for _ in range(N)])

        if with_pe:
            if self.with_pe == 'rpe':
                # self.fc_g = nn.Linear(64, h*d_k)
                # self.h = h
                # self.d_k = d_k
                self.rddpe = PolarRPE(k=3, h=h, d_k=d_k, d_r=256, device=device)
            else:
                self.grid_embedding = GridPESine(d_model/2, normalize=True)

    def forward(self, input):
        # input (b_s, seq_len, d_in)
        b_s = input.shape[0]
        out = self.in_proj_model(input)

        geometric_attention = None
        grid_pos = None
        if self.with_pe:
            if self.with_pe == 'rpe':
                # g = RelationalEmbedding(input, is_gird=True)
                # b_s, n = g.shape[:2]
                # g = self.fc_g(g.view(-1, 64))
                # g = g.view(b_s, n, n, self.h, self.d_k).permute(0, 3, 1, 2, 4)
                # geometric_attention = F.relu(g)  # (b_s, h, n, n, d_k)
                geometric_attention = self.rddpe(b_s)
            else:
                grid_pos = self.grid_embedding(out.view(b_s, 9, 9, -1))

        if self.multi_level:
            outs = []
            for l in self.layers:
                out = l(out, out, out, geometric_attention=geometric_attention, grid_pos=grid_pos)
                outs.append(out.unsqueeze(1))
            out = torch.cat(outs, 1)
            
        else:
            for i, l in enumerate(self.layers):
                out = l(out, out, out, geometric_attention=geometric_attention, grid_pos=grid_pos)

        return out


def build_encoder(N, d_in=2560, with_pe=None, mid_dim=40, multi_level=False, local=False, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
    Encoder = TransformerEncoder(N, d_in, d_model, d_k, d_v, h, d_ff, dropout, with_pe, mid_dim, multi_level, local, device)
    
    return Encoder