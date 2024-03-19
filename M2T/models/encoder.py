import torch
from torch import nn
from common.models.transformer.encoders import EncoderLayer, TransformerEncoder
from common.models.transformer.attention import ScaledDotProductAttentionMemory


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask


class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, m, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, d_model, d_k, d_v, h, d_ff, dropout, attention_module = ScaledDotProductAttentionMemory, 
                                                     attention_module_kwargs={'m': m})
        self.in_proj_model = nn.Sequential(
            nn.Linear(d_in, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

    def forward(self, input, attention_weights=None):
        out = self.in_proj_model(input)
        return super(MemoryAugmentedEncoder, self).forward(out, attention_weights=attention_weights)


# def build_encoder(N, padding_idx, m, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
#     return MemoryAugmentedEncoder(N, padding_idx, m, d_in, d_model, d_k, d_v, h, d_ff, dropout)

def build_encoder(N, padding_idx, m, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
    return TransformerEncoder(N, padding_idx, d_in, d_model, d_k, d_v, h, d_ff, 
                              dropout, attention_module = ScaledDotProductAttentionMemory, 
                              attention_module_kwargs={'m': m})

