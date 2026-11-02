import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from Transformer import *
import copy

# 定义AVCE_Model类，继承自nn.Module，用于构建一个多模态的神经网络模型
class AVCE_Model(nn.Module):
    # 初始化函数，接收参数args
    def __init__(self, args):
        # 调用父类的初始化方法
        super(AVCE_Model, self).__init__()
        # 定义一个深拷贝函数，用于后续复制某些模块
        c = copy.deepcopy
        # 从参数args中获取dropout比例
        dropout = args.dropout
        # 从参数args中获取多头注意力的头数
        nhead = args.nhead
        # 从参数args中获取隐藏层的维度
        hid_dim = args.hid_dim
        # 从参数args中获取前馈网络的维度
        ffn_dim = args.ffn_dim
        # 初始化一个多头注意力模块
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        # 初始化一个位置前馈网络模块
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        # 定义一个全连接层，将1024维的视觉特征映射到hid_dim维
        self.fc_v = nn.Linear(1024, hid_dim)
        # 定义一个全连接层，将128维的音频特征映射到hid_dim维
        self.fc_a = nn.Linear(128, hid_dim)
        # 初始化一个交叉注意力块，其中包含一个Transformer层
        self.cma = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), c(self.feedforward), dropout))
        # 初始化一个Att_MMIL模块，用于多模态交互
        self.att_mmil = Att_MMIL(hid_dim, args.num_classes)
        # ================= 新增：真实 CLIP 静态语义锚点 =================
        try:
            anchor_abn = torch.load('anchor_abn_clip.pt')
            anchor_nor = torch.load('anchor_nor_clip.pt')

            # 【核心绝杀】：消除 CLIP 文本中 "A video of..." 带来的极高相似度
            # 保持 异常锚点 方向绝对不变，将 正常锚点 强制投影到与其绝对垂直(90度)的空间上！
            # 这样既利用了 CLIP 的语义轴，又拿到了最大的对比 Margin！
            dot_product = torch.sum(anchor_abn * anchor_nor, dim=-1, keepdim=True)
            anchor_nor_ortho = anchor_nor - dot_product * anchor_abn
            anchor_nor_ortho = F.normalize(anchor_nor_ortho, p=2, dim=-1)

            anchor_nor = anchor_nor_ortho  # 替换为正交化后的正常锚点
            print(">>> 成功加载 CLIP 文本先验并完成施密特正交化分离！ <<<")

        except FileNotFoundError:
            print(">>> 警告: 未找到 CLIP 锚点，使用随机备用锚点 <<<")
            anchor_abn = torch.empty(1, args.hid_dim)
            anchor_nor = torch.empty(1, args.hid_dim)
            nn.init.orthogonal_(anchor_abn)
            nn.init.orthogonal_(anchor_nor)

        self.register_buffer('anchor_abn', anchor_abn)
        self.register_buffer('anchor_nor', anchor_nor)
        # ================================================================
    # 前向传播函数，接收音频特征f_a、视觉特征f_v和序列长度seq_len
    def forward(self, f_a, f_v, seq_len):
        # 将视觉特征和音频特征分别通过全连接层进行维度变换
        f_v, f_a = self.fc_v(f_v), self.fc_a(f_a)
        # 将变换后的视觉特征和音频特征输入到交叉注意力块中，得到输出
        v_out, a_out = self.cma(f_v, f_a)
        # 将交叉注意力块的输出输入到Att_MMIL模块中，得到不同的logits
        mmil_logits, audio_logits, visual_logits, av_logits = self.att_mmil(a_out, v_out, seq_len)
        # 返回各种logits以及交叉注意力块的输出
        return mmil_logits, audio_logits, visual_logits, av_logits, v_out, a_out, self.anchor_abn, self.anchor_nor

# 定义Att_MMIL类，继承自nn.Module，用于多模态交互
class Att_MMIL(nn.Module):
    # 初始化函数，接收输入维度input_dim和类别数num_classes
    def __init__(self, input_dim, num_classes):
        # 调用父类的初始化方法
        super(Att_MMIL, self).__init__()
        # 定义一个全连接层，将输入维度映射到类别数
        self.fc = nn.Linear(input_dim, num_classes)

    # 分类函数，用于对logits进行处理
    def clas(self, logits, seq_len):
        # 去除logits的单维度
        logits = logits.squeeze()
        # 初始化一个空的张量，用于存储实例级别的logits
        instance_logits = torch.zeros(0).cuda()  # tensor([])
        # 遍历每个样本
        for i in range(logits.shape[0]):
            # 如果序列长度seq_len为None
            if seq_len is None:
                # 计算当前样本logits的均值，并调整形状
                tmp = torch.mean(logits[i]).view(1)
            else:
                # 从当前样本的logits中选取前seq_len[i]个元素，并取前seq_len[i] // 16 + 1个最大值
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
                # 计算这些最大值的均值，并调整形状

                tmp = torch.mean(tmp).view(1)
            # 将当前样本的处理结果拼接到instance_logits中
            instance_logits = torch.cat((instance_logits, tmp))
        # 对instance_logits应用sigmoid函数
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    # 前向传播函数，接收音频输出a_out、视觉输出v_out和序列长度seq_len
    def forward(self, a_out, v_out, seq_len):
        # prediction
        # 确保两个张量具有相同的序列长度，选择较小的序列长度

        # 在新的维度上拼接音频和视觉输出
        x = torch.cat([a_out.unsqueeze(-2), v_out.unsqueeze(-2)], dim=-2)
        # 将拼接后的结果通过全连接层
        frame_prob = self.fc(x)
        # 在维度2上求和得到av_logits
        av_logits = frame_prob.sum(dim=2)
        # 对frame_prob的音频部分应用sigmoid函数
        a_logits = torch.sigmoid(frame_prob[:, :, 0, :])
        # 对frame_prob的视觉部分应用sigmoid函数
        v_logits = torch.sigmoid(frame_prob[:, :, 1, :])
        # 调用clas函数对av_logits进行处理
        mmil_logits = self.clas(av_logits, seq_len)
        # 返回各种logits
        return mmil_logits, a_logits, v_logits, av_logits

# 定义Single_Model类，继承自nn.Module，用于构建单模态的神经网络模型
class Single_Model(nn.Module):
    # 初始化函数，接收参数args
    def __init__(self, args):
        # 调用父类的初始化方法
        super(Single_Model, self).__init__()
        # 定义一个深拷贝函数，用于后续复制某些模块
        c = copy.deepcopy
        # 从参数args中获取dropout比例
        dropout = args.dropout
        # 从参数args中获取多头注意力的头数
        nhead = args.nhead
        # 从参数args中获取隐藏层的维度
        hid_dim = args.hid_dim
        # 从参数args中获取前馈网络的维度
        ffn_dim = args.ffn_dim
        # 从参数args中获取视觉特征的维度
        n_dim = args.v_feature_size
        # 初始化一个多头注意力模块
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        # 初始化一个位置前馈网络模块
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        # 定义一个全连接层，将n_dim维的特征映射到hid_dim维
        self.fc_v = nn.Linear(n_dim, hid_dim)
        # 初始化一个自注意力块，其中包含一个Transformer层
        self.cma = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), c(self.feedforward), dropout))
        # 定义一个全连接层，将hid_dim维的特征映射到类别数
        self.fc = nn.Linear(hid_dim, args.num_classes)

    # 分类函数，用于对logits进行处理
    def clas(self, logits, seq_len):
        # 去除logits的单维度
        logits = logits.squeeze()
        # 初始化一个空的张量，用于存储实例级别的logits
        instance_logits = torch.zeros(0).cuda()  # tensor([])
        # 遍历每个样本
        for i in range(logits.shape[0]):
            # 从当前样本的logits中选取前seq_len[i]个元素，并取前seq_len[i] // 16 + 1个最大值
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            # 计算这些最大值的均值，并调整形状
            tmp = torch.mean(tmp).view(1)
            # 将当前样本的处理结果拼接到instance_logits中
            instance_logits = torch.cat((instance_logits, tmp))
        # 对instance_logits应用sigmoid函数
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    # 前向传播函数，接收特征f和序列长度seq_len
    def forward(self, f, seq_len):
        # 将特征f通过全连接层进行维度变换
        f = self.fc_v(f)
        # 将变换后的特征输入到自注意力块中，得到输出
        sa = self.cma(f)
        # 将自注意力块的输出通过全连接层
        out = self.fc(sa)
        # 如果序列长度seq_len不为None
        if seq_len is not None:
            # 调用clas函数对out进行处理
            out = self.clas(out, seq_len)
        # 返回处理后的结果
        return out