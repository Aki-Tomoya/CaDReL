import torch
from torch import nn
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


class TransformerEncoder(nn.Module):
    def __init__(self, N, padding_idx = None, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level = False,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None, iscamel = True):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.multi_level = multi_level
        self.iscamel = iscamel
        self.att_outs = []
        self.in_proj_model = nn.Sequential(
            nn.Linear(d_in, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

    def forward(self, input, attention_weights=None, pos = None):
        # input (b_s, seq_len, d_in)
        attention_mask = None
        if self.padding_idx is not None:
            # (b_s, 1, 1, seq_len)
            attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

        # print(self.in_proj_model)
        # print(input.shape)
        out = self.in_proj_model(input)
        # try:
        #     out = self.in_proj_model(input)
        # except RuntimeError as e:
        #     print(self.in_proj_model)
        #     print(input.shape)

        if self.multi_level:
            outs = []
            for l in self.layers:
                out = l(out, out, out, attention_mask, attention_weights)
                outs.append(out.unsqueeze(1))

            outs = torch.cat(outs, 1)
            return outs, attention_mask

# /w\ 保存注意力图在att_outs中
        if self.iscamel:
            outs = []
            att_outs = []
            for l in self.layers:
                out, att_out = l(out, out, out, attention_mask, attention_weights)
                # outs.append(out.unsqueeze(1))
                # att_outs = torch.cat(att_outs, att_out.unsqueeze(1))

                # att_outs.append(att_out.unsqueeze(1))
                att_outs.append(att_out.unsqueeze(1))
            
            # att_outs = torch.cat(att_outs, 1)
            att_outs = torch.cat(att_outs, 1)
            # outs = torch.cat(outs, 1)

            return out, attention_mask, att_outs
            # return outs, attention_mask

        else:
            for l in self.layers:
                out = l(out, out, out, attention_mask, attention_weights, pos)

            return out, attention_mask