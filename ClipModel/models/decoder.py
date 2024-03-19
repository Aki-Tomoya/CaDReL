import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common.models.containers import Module, ModuleList
from common.models.transformer.attention import MultiHeadAttention
from common.models.transformer.utils import PositionWiseFeedForward, sinusoid_encoding_table

class MeshedDecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level=False, mid_dim=128):
        super(MeshedDecoderLayer, self).__init__()
        self.multi_level = multi_level
        self.d_v = d_v
        self.d_k = d_k
        self.h = h

        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        if multi_level:
            self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

            # self.in_proj_q = nn.Sequential(
            #     nn.Linear(d_model, h * d_k),
            #     nn.ELU(),
            #     nn.GroupNorm(h, h * d_k)
            # )

            # self.in_proj_k = nn.Sequential(
            #     nn.Linear(d_model, h * d_k),
            #     nn.ELU(),
            #     nn.GroupNorm(h, h * d_k)
            # )

            # self.in_proj_v = nn.Sequential(
            #     nn.Linear(d_model, h * d_v),
            #     nn.ELU(),
            #     nn.GroupNorm(h, h * d_v)
            # )

            # self.proj_attn_map = nn.Sequential(
            #     nn.Linear(d_k, mid_dim),
            #     nn.ReLU(),
            #     nn.Dropout(dropout)
            # )

            # self.proj_attn_spatial = nn.Sequential(
            #     nn.Linear(mid_dim, 1),
            #     nn.Sigmoid()
            # )

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        if self.multi_level:
            enc_att1 = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att) * mask_pad
            enc_att2 = self.enc_att(self_att, enc_output[:, 1], enc_output[:, 1], mask_enc_att) * mask_pad
            enc_att3 = self.enc_att(self_att, enc_output[:, 2], enc_output[:, 2], mask_enc_att) * mask_pad

            alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
            alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
            alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))
            enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3) / np.sqrt(3)

            # bs, nq = self_att.shape[:2]

            # self_att = self_att.view(bs * nq, -1)
            # self_att_q = self.in_proj_q(self_att).view(bs, nq, self.h, self.d_k).transpose(1, 2) # (b_s, h, nq, d_k)

            # v_arr = []
            # alpha_arr = []
            # for enc_att in [enc_att1, enc_att2, enc_att3]:
            #     enc_att = enc_att.view(bs * nq, -1)

            #     enc_att_k = self.in_proj_k(enc_att).view(bs, nq, self.h, self.d_k).transpose(1, 2) # (b_s, h, nq, d_k)
            #     enc_att_v = self.in_proj_v(enc_att).view(bs, nq, self.h, self.d_k).transpose(1, 2) # (b_s, h, nq, d_k)

            #     attn_map = self.proj_attn_map(self_att_q * enc_att_k)
            #     alpha = self.proj_attn_spatial(attn_map)

            #     v_arr.append(enc_att_v)
            #     alpha_arr.append(alpha)

            # enc_att = (v_arr[0] * alpha_arr[0] + v_arr[1] * alpha_arr[1] + v_arr[2] * alpha_arr[2]) / np.sqrt(3)

            # enc_att = enc_att.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_v)
        else:
            enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)

        enc_att = enc_att * mask_pad
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff

class TransformerDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, multi_level=False, aux_loss=False, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model

        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)

        self.layers = ModuleList([MeshedDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, multi_level) for _ in range(N_dec)])

        if aux_loss:
            self.fcs = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in range(N_dec)])
        else:
            self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.aux_loss = aux_loss
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder=None):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1)  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)

        if self.aux_loss:
            outs = []
            for l, fc in zip(self.layers, self.fcs):
                out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)
                
                x = fc(out)
                x = F.log_softmax(x, dim=-1)
                outs.append(x)

            return (outs[0] + outs[1] + outs[2]) / 3

        else:
            for l in self.layers:
                out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

            out = self.fc(out)
            return F.log_softmax(out, dim=-1)

def build_decoder(vocab_size, max_len, N_dec, padding_idx, multi_level=False, aux_loss=False,
                  d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):

    Decoder = TransformerDecoder(vocab_size, max_len, N_dec, padding_idx, multi_level, aux_loss,
                               d_model, d_k, d_v, h, d_ff, dropout)
                               
    return Decoder