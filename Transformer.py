import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

# 定义自注意力块类
class SelfAttentionBlock(nn.Module):
    # 初始化函数，接收一个注意力层作为参数
    def __init__(self, attention_layer):
        # 调用父类的初始化方法
        super(SelfAttentionBlock, self).__init__()
        # 保存传入的注意力层
        self.layer = attention_layer
        # 保存注意力层的大小
        self.size = attention_layer.size

    # 前向传播函数，接收特征作为输入
    def forward(self, feature):
        # 将特征作为查询、键和值输入到注意力层中，得到自注意力处理后的特征
        feature_sa = self.layer(feature, feature, feature)
        # 返回自注意力处理后的特征
        return feature_sa

# 定义交叉注意力块类
class CrossAttentionBlock(nn.Module):
    # 初始化函数，接收一个注意力层作为参数
    def __init__(self, attention_layer):
        # 调用父类的初始化方法
        super(CrossAttentionBlock, self).__init__()
        # 保存传入的注意力层
        self.layer = attention_layer
        # 保存注意力层的大小
        self.size = attention_layer.size

    # 前向传播函数，接收视频特征和音频特征作为输入
    def forward(self, video, audio):
        # 将视频特征作为查询，音频特征作为键和值，输入到注意力层中，得到视频的交叉注意力处理结果
        video_cma = self.layer(video, audio, audio)
        # 将音频特征作为查询，视频特征作为键和值，输入到注意力层中，得到音频的交叉注意力处理结果
        audio_cma = self.layer(audio, video, video)
        # 返回视频和音频的交叉注意力处理结果
        return video_cma, audio_cma

# 定义Transformer层类
class TransformerLayer(nn.Module):
    # 初始化函数，接收大小、自注意力模块、前馈神经网络模块和丢弃率作为参数
    def __init__(self, size, self_attn, feed_forward, dropout):
        # 调用父类的初始化方法
        super(TransformerLayer, self).__init__()
        # 保存自注意力模块
        self.self_attn = self_attn
        # 保存前馈神经网络模块
        self.feed_forward = feed_forward
        # 创建两个子层连接模块，用于残差连接
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 保存层的大小
        self.size = size

    # 前向传播函数，接收查询、键和值作为输入
    def forward(self, q, k, v):
        # 首先将查询输入到第一个子层连接模块中，经过自注意力处理，得到更新后的查询
        q = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[0])
        # 将更新后的查询输入到第二个子层连接模块中，经过前馈神经网络处理，得到最终输出
        return self.sublayer[1](q, self.feed_forward)

# 定义子层连接类，用于实现残差连接
class SublayerConnection(nn.Module):
    # 初始化函数，接收层的大小和丢弃率作为参数
    def __init__(self, size, dropout):
        # 调用父类的初始化方法
        super(SublayerConnection, self).__init__()
        # 创建层归一化模块，用于对输入进行归一化
        self.norm = nn.LayerNorm(size)
        # 创建丢弃模块，用于在训练时随机丢弃部分神经元
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数，接收输入和子层函数作为参数
    def forward(self, x, sublayer):
        # 对输入进行归一化，然后通过子层函数处理，再经过丢弃操作，最后与原始输入相加，实现残差连接
        return x + self.dropout(sublayer(self.norm(x)))

# 定义注意力计算函数，接收查询、键、值、掩码大小和丢弃率作为参数
def attention(query, key, value, masksize, dropout=None):
    # 计算查询的最后一个维度的大小，即特征维度
    d_k = query.size(-1)
    # 计算查询和键的点积，并除以特征维度的平方根，得到注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果掩码大小不等于1，说明需要使用掩码
    if masksize != 1:
        # 将掩码大小减半
        masksize = int(masksize / 2)
        # 创建一个与注意力分数相同大小的全1张量，作为掩码
        mask = torch.ones(scores.size()).cuda()
        # 遍历掩码的第三个维度
        for i in range(mask.shape[2]):
            # 如果当前位置减去掩码大小大于0，将掩码的相应部分置为0
            if i - masksize > 0:
                mask[:, :, i, :i - masksize] = 0
            # 如果当前位置加上掩码大小加1小于掩码的第四个维度，将掩码的相应部分置为0
            if i + masksize + 1 < mask.shape[3]:
                mask[:, :, i, masksize + i + 1:] = 0
        # 打印掩码的第一个样本的第一个头的信息（调试用）
        # print(mask[0][0])
        # 将注意力分数中掩码为0的位置填充为一个非常小的值，以避免对softmax计算产生影响
        scores = scores.masked_fill(mask == 0, -1e9)
    # 对注意力分数进行softmax操作，得到注意力权重
    p_attn = F.softmax(scores, dim=-1)
    # 如果丢弃率不为空，对注意力权重进行丢弃操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 将注意力权重与值相乘，得到加权后的特征
    return torch.matmul(p_attn, value), p_attn

# 定义克隆模块函数，用于复制模块N次
def clones(module, N):
    # 使用深拷贝创建N个模块，并将它们存储在ModuleList中
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 定义多头注意力类
class MultiHeadAttention(nn.Module):
    # 初始化函数，接收头的数量、模型维度、掩码大小和丢弃率作为参数
    def __init__(self, h, d_model, masksize=1, dropout=0.1):
        # 调用父类的初始化方法
        super(MultiHeadAttention, self).__init__()
        # 确保模型维度可以被头的数量整除
        assert d_model % h == 0
        # 计算每个头的特征维度
        self.d_k = d_model // h
        # 保存头的数量
        self.h = h
        # 创建4个线性层，用于对查询、键、值进行线性变换
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 初始化注意力权重为None
        self.attn = None
        # 保存掩码大小
        self.masksize = masksize
        # 创建层归一化模块
        self.layer_norm = nn.LayerNorm(d_model)
        # 创建丢弃模块
        self.dropout = nn.Dropout(p=dropout)

    # 前向传播函数，接收查询、键和值作为输入
    def forward(self, query, key, value):
        # 获取查询的批量大小
        nbatches = query.size(0)

        # 1) 对查询、键和值进行线性投影，将维度从d_model转换为h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 2) 对投影后的向量应用注意力机制
        x, self.attn = attention(query, key, value, self.masksize, dropout=self.dropout)
        # 3) 将多头的结果拼接起来，并通过最后一个线性层进行变换
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)
        # 返回最终输出和注意力权重
        return out, self.attn

# 定义位置前馈神经网络类
class PositionwiseFeedForward(nn.Module):
    # 初始化函数，接收模型维度、前馈网络维度和丢弃率作为参数
    def __init__(self, d_model, d_ff, dropout=0.1):
        # 调用父类的初始化方法
        super(PositionwiseFeedForward, self).__init__()
        # 创建第一个线性层，将输入维度从d_model转换为d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # 创建第二个线性层，将输入维度从d_ff转换回d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # 创建丢弃模块
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数，接收输入作为参数
    def forward(self, x):
        # 先通过第一个线性层，然后应用ReLU激活函数，再进行丢弃操作，最后通过第二个线性层，得到输出
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        # 返回输出
        return output