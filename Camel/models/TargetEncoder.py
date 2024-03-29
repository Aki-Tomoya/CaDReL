import torch
from torch import nn
from common.models.transformer.attention import MultiHeadAttention
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
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights=attention_weights)
        ff = self.pwff(att)
        return ff, att

# Skipping Frequency Diffusion 

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


 


def SFDModule(self, LowHS, HighHS, attention_mask=None, attention_weights=None, pos=None):
    ##特征向量进行归一化
    for i in LowHS.size():
        for j in HighHS.size():
            LowHS[i], HighHS[j]=normalize(LowHS[i]),normalize(HighHS[j])
            cos=1-torch.mm(LowHS[i], HighHS[j].permute(1, 0))
            cos.shape()

class DiffusionSection(nn.Module):
    def __init__(self, N, padding_idx = None, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level = False,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(TargetEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.multi_level = multi_level


        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)]
                                    [EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)]
                                    # [SFDModule(d_model, hiddenstat1, hiddenstat2,
                                    #               identity_map_reordering=identity_map_reordering,
                                    #               attention_module=attention_module,
                                    #               attention_module_kwargs=attention_module_kwargs)]
                                    )
    

class TargetEncoder(nn.Module):
    def __init__(self, N, padding_idx = None, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level = False,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None, iscamel = False):
        super(TargetEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.multi_level = multi_level
        self.iscamel = iscamel

        self.in_proj_model = nn.Sequential(
            nn.Linear(d_in, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

        self.layers = nn.ModuleList([DiffusionSection(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(2)])

    def forward(self, input, attention_weights=None, pos = None):
        # input (b_s, seq_len, d_in)
        attention_mask = None
        if self.padding_idx is not None:
            # (b_s, 1, 1, seq_len)
            attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

        out = self.in_proj_model(input)

        if self.multi_level:
            outs = []
            for l in self.layers:
                out = l(out, out, out, attention_mask, attention_weights)
                outs.append(out.unsqueeze(1))

            outs = torch.cat(outs, 1)
            return outs, attention_mask

            # /w\
        if self.iscamel:
            att_outs = []
            outs = []
            for l in self.layers:
                out, att_outs = l(out, out, out, attention_mask, attention_weights)
                outs.append(out.unsqueeze(1))
                att_outs.append(att_outs.unsqueeze(1))
            
            att_outs = torch.cat(att_outs, 1)
            outs = torch.cat(outs, 1)
            return outs, attention_mask, att_outs

        else:
            for l in self.layers:
                out = l(out, out, out, attention_mask, attention_weights, pos)

            return out, attention_mask