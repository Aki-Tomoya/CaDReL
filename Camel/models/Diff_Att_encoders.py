import math
import torch
from torch import nn
import torch.nn.functional as F
from Camel.utils.attention import MultiHeadAttention
from common.models.transformer.utils import PositionWiseFeedForward
import torch.autograd as autograd
import numpy as np
autograd.set_detect_anomaly(True)


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

class SkippingDiffusionLayer(nn.Module):
    def __init__(self):
        super(SkippingDiffusionLayer, self).__init__()
    

    def forward(self, LowLevelFeatures, HighLevelFeatures, batch_size):
        
        btx = batch_size

        LLF, HLF = LowLevelFeatures.clone().squeeze(1), HighLevelFeatures.clone().squeeze(1)
        # 生成相似度矩阵
        # similarity_weight = adjusted_cosine_similarity(LLF.unsqueeze(2), HLF.unsqueeze(1))
        similarity = torch.cosine_similarity(LLF.unsqueeze(2), HLF.unsqueeze(1), dim=-1)
        # similarity = torch.cosine_similarity(LLF, HLF, dim=-1)
        similarity = similarity / math.sqrt(LLF.size(-1))

        # attention_probs = similarity
        attention_probs = torch.softmax(similarity / 1 ,dim=1)

        # Test
        # numpy_array1 = similarity.to('cpu').detach().numpy()
        # numpy_array2 = attention_probs.to('cpu').detach().numpy()
        # np.save('/home/zkx/ImgCap/chart/similarity_matrix.npy', numpy_array1)
        # np.save('/home/zkx/ImgCap/chart/attetion_matrix.npy', numpy_array2)

        # 使用gumblesoftmax代替argmax,其函数输出为独热向量
        #TODO:3
        gumbel_output = F.gumbel_softmax(attention_probs, tau=3, hard=True, eps=1e-10, dim=1)

        # 从 one-hot 向量中找出每行最大值的位置
        one_hot_vectors = gumbel_output

        # 获取每行最大值的位置，每个低级特征对应高级特征的序列
        # similarity_list = torch.nonzero(one_hot_vectors, as_tuple=False)[:, 1]
        similarity_list = torch.where(one_hot_vectors > 0.5)[1]
        
        # reshape
        similarity_list = similarity_list.reshape(-1, LLF.size(1))
        
        # 构建索引张量
        index_tensor = similarity_list.unsqueeze(-1).expand(-1, -1, HLF.size(-1)).long()
        # index_tensor = similarity_list.unsqueeze(1).expand(-1, LLF.size(1), -1).long()

        # 使用gather函数将LLF对齐到HLF上
        LLF_aligned = torch.gather(HLF, 1, index_tensor)

        # 计算相似度权重
        # [20,144,512]
        similarity_weight = torch.cosine_similarity(LLF_aligned, LLF, dim=-1)

        # 将相似度权重合并成一个张量
        similarity_list_feature = similarity_weight

        similarity_list_feature = similarity_list_feature.reshape(-1, LLF.size(1))

        noise_parameter = torch.softmax(similarity_list_feature / 0.1, dim=1)

        # Method1:将对应低级特征直接加到高级特征上去

        # 添加两个额外的维度以匹配LLF张量的形状
        noise_parameter = noise_parameter.unsqueeze(-1)

        # 添加两个额外的维度以匹配HLF张量的形状
        similarity_list = similarity_list.unsqueeze(-1).expand(-1, -1, HLF.size(-1))

        # 计算temp张量，使用*运算符进行逐元素相乘
        temp = LLF * noise_parameter

        # .contiguous()
        # 使用gather函数从HLF张量中收集相应条目，并计算HLF_new张量
        HLF_new = torch.gather(HLF, 1, similarity_list) + temp

        # 使用scatter函数将HLF_new张量分配回HLF张量中，更新HLF张量的相应条目
        HLF = HLF.scatter(1, similarity_list, HLF_new)
        
        out = HLF 

        return out



class TransformerEncoder(nn.Module):

    def __init__(self, N, padding_idx = None, d_in=2048, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level = False,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None, iscamel=True, batch_size=None, isMesh=None):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.multi_level = multi_level
        self.iscamel = iscamel
        self.isMesh = isMesh
        self.btx = batch_size
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
        self.diffusion_layer = SkippingDiffusionLayer()

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
                outs.append(out.unsqueeze(1))
                att_outs.append(att_out.unsqueeze(1))

            att_outs = torch.cat(att_outs, 1)

            # Diffusion计算
            diffout = self.diffusion_layer(outs[0], outs[-1], self.btx)
            out = diffout.squeeze(1)


            return out, attention_mask, att_outs
            # return outs, attention_mask

        else:
            for l in self.layers:
                out = l(out, out, out, attention_mask, attention_weights, pos)

            return out, attention_mask
        


def adjusted_cosine_similarity(x1, x2):
    x1_mean = x1.mean(dim=1, keepdim=True)
    x2_mean = x2.mean(dim=1, keepdim=True)
    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean
    numerator = (x1_centered * x2_centered).sum(dim=1).sum(dim=1)
    denominator = torch.norm(x1_centered) * torch.norm(x2_centered)
    return numerator / denominator