from common.models.transformer.attention import AoAttention, MemoryAttention, ScaledDotProductAttentionMemory
from common.models.transformer.encoders import TransformerEncoder

def build_encoder(N, padding_idx, d_in=2048, attention_module=None, attention_module_kwargs=None, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
    Encoder = TransformerEncoder(N, padding_idx, d_in, d_model, d_k, d_v, h, d_ff, dropout, 
                                 attention_module=attention_module, attention_module_kwargs=attention_module_kwargs)
    
    return Encoder