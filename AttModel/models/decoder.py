from common.models.transformer.decoders import TransformerDecoder
from common.models.transformer.attention import AoAttention


def build_decoder(vocab_size, max_len, N_dec, padding_idx,  
                  d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, **kwargs):
    # aoADecoder = TransformerDecoder(vocab_size, max_len, N_dec, padding_idx,  
    #                            d_model, d_k, d_v, h, d_ff, dropout, 
    #                            self_att_module=AoAttention, enc_att_module=AoAttention)
    Decoder = TransformerDecoder(vocab_size, max_len, N_dec, padding_idx,  
                               d_model, d_k, d_v, h, d_ff, dropout)
                               
    return Decoder