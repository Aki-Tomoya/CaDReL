import torch
from torch import nn
from torch.nn import functional as F
from common.models.containers import Module, ModuleList
from common.models.transformer.attention import MultiHeadAttention
from common.models.transformer.utils import PositionWiseFeedForward, sinusoid_encoding_table

class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True)
        self.cross_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_out, mask_pad, mask_self_att, mask_enc_att=None):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        enc_att = self.cross_att(self_att, enc_out, enc_out, mask_enc_att)
        enc_att = enc_att * mask_pad
        
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff

class TransformerDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, return_logits = False):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model

        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec
        self.return_logits = return_logits

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, enc_out, mask_enc=None):

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
        for i, l in enumerate(self.layers):
            out = l(out, enc_out, mask_queries, mask_self_attention, mask_enc)

        out = self.fc(out)
        if self.return_logits:
            return out
        else:
            return F.log_softmax(out, dim=-1)

def build_decoder(vocab_size, max_len, N_dec, padding_idx,  
                  d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, **kwargs):
    Decoder = TransformerDecoder(vocab_size, max_len, N_dec, padding_idx,  
                               d_model, d_k, d_v, h, d_ff, dropout)
                               
    return Decoder