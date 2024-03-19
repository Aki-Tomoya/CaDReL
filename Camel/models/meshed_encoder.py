import torch
from torch import nn
# from common.models.transformer.encoders import EncoderLayer, TransformerEncoder
from common.models.transformer.encoders import TransformerEncoder
# from common.models.transformer.attention import ScaledDotProductAttentionMemory
from Camel.utils.attention import ScaledDotProductAttentionMemory
from Camel.models.Diff_Att_encoders import SkippingDiffusionLayer
from Camel.utils.attention import MultiHeadAttention
from common.models.transformer.utils import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()

        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, pos=None):
        if pos is not None:
            queries = queries + pos
            keys = keys + pos
        att, att_out = self.mhatt(queries, keys, values, attention_mask, attention_weights=attention_weights)
        ff = self.pwff(att)
        # /w\
        # return ff
        return ff, att_out


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None, iscamel=True, batch_size=None, isMesh=True):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.iscamel = iscamel
        self.isMesh = isMesh
        self.btx = batch_size
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.diffusion_layer = SkippingDiffusionLayer()

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        outs = []
        att_outs = []
        out = input
        for l in self.layers:
            out, att_out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))
            att_outs.append(att_out.unsqueeze(1))

        att_outs = torch.cat(att_outs, 1)
        outs = torch.cat(outs, 1)
        diffout = self.diffusion_layer(outs[0], outs[-1], self.btx)
        out = diffout.squeeze(1)

        return out, attention_mask, att_outs, outs


class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, m, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, iscamel=True, batch_size=None, isMesh=True):
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

