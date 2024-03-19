import torch
from torch import nn
from torch.nn import functional as F
from common.models.transformer.attention import MultiHeadAttention, NormSelfAttention
from common.models.transformer.utils import PolarRPE, PositionWiseFeedForward, RelationalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, with_pe=None):
        super(EncoderLayer, self).__init__()

        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout,
                                        attention_module=NormSelfAttention,
                                        attention_module_kwargs={'with_pe': with_pe})

        # self.self_att2 = MultiHeadAttention(d_model, d_k, d_v, h, dropout)

        self.cross_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        # self.cross_att2 = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        
        self.self_pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.cross_pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    # def forward(self, gird_features, region_features, object_queries, attention_mask, geometric_attention=None):
    def forward(self, gird_features, region_features, attention_mask, geometric_attention=None):

        grid_att = self.self_att(gird_features, gird_features, gird_features, attention_weights=geometric_attention)

        # region_features = self.self_att2(region_features, region_features, region_features, attention_mask=attention_mask)
        object_att = self.cross_att(region_features, region_features, region_features, attention_mask=attention_mask)

        # object_att = self.cross_att(object_queries, region_features, region_features, attention_mask=attention_mask)
        # object_att = self.self_att2(object_att, object_att, object_att)

        gird_ff = self.self_pwff(grid_att)
        object_ff = self.cross_pwff(object_att)
        # return gird_ff, object_ff
        return gird_ff, object_ff

class TransformerEncoder(nn.Module):
    def __init__(self, N, prefix_length=10, with_pe=None, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.with_pe = with_pe
        self.dim_g = 64

        self.grid_proj = nn.Sequential(
            nn.Linear(2048, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

        self.region_proj = nn.Sequential(
            nn.Linear(2048, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

        if with_pe and with_pe == 'rpe':
            self.fc = nn.Linear(4, self.dim_g)
            self.act = nn.ReLU()
            self.fc_g = nn.Linear(self.dim_g, h*d_k)
            self.h = h
            self.d_k = d_k
            # self.rddpe = PolarRPE(k=1, h=h, d_k=d_k, d_r=64, device=device)

        # self.object_queries = nn.Parameter(torch.randn(prefix_length, d_model), requires_grad=True)

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, with_pe) for _ in range(N)])

    def forward(self, grid_features, region_features):
        # input (b_s, seq_len)
        attention_mask = (torch.sum(torch.abs(region_features), -1) == 0).unsqueeze(1).unsqueeze(1)
        grid_features = self.grid_proj(grid_features)
        region_features = self.region_proj(region_features)

        b_s = region_features.shape[0]
        # query_shape = self.object_queries.shape
        # object_queries = self.object_queries.unsqueeze(0).expand(b_s, *query_shape).contiguous()

        geometric_attention = None
        if self.with_pe and self.with_pe == 'rpe':
            g = RelationalEmbedding(grid_features, dim_g=self.dim_g, is_gird=True, trignometric_embedding=False)
            b_s, n = g.shape[:2]
            g = self.act(self.fc(g.view(-1, 4)))
            g = self.fc_g(g)
            # g = self.fc_g(g.view(-1, self.dim_g))
            g = g.view(b_s, n, n, self.h, self.d_k).permute(0, 3, 1, 2, 4)
            geometric_attention = F.relu(g)  # (b_s, h, n, n, d_k)
            # geometric_attention = self.rddpe(b_s)

        for l in self.layers:
            # grid_features, object_queries = l(grid_features, region_features, object_queries, attention_mask, geometric_attention)
            grid_features, region_features = l(grid_features, region_features, attention_mask, geometric_attention)
            # grid_features, region_features = l(grid_features, region_features, attention_mask)

        # out = torch.cat([grid_features, region_features], dim=1)

        # add_mask = torch.zeros(b_s, 1, 1, grid_features.shape[1]).bool().to(region_features.device)
        # attention_mask = torch.cat([add_mask, attention_mask], dim=-1)

        # return out, attention_mask
        # return grid_features, object_queries
        return grid_features, region_features, attention_mask

def build_encoder(N, prefix_length=10, with_pe=None, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
    Encoder = TransformerEncoder(N, prefix_length, with_pe, device, d_model, d_k, d_v, h, d_ff, dropout)
    
    return Encoder