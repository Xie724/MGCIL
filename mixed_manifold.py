import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedManifoldInfoNCE(nn.Module):
    def __init__(self, feature_dim, temperature=0.1):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature]))

        self.need_proj = (feature_dim % 3 != 0)
        if self.need_proj:
            target_dim = (feature_dim // 3) * 3
            self.proj = nn.Linear(feature_dim, target_dim)
            nn.init.xavier_uniform_(self.proj.weight, gain=0.1)

    def forward(self, query, positive_key, negative_keys):
        if self.need_proj:
            query = self.proj(query)
            positive_key = self.proj(positive_key)
            negative_keys = self.proj(negative_keys)

        # ================= 🚀 终极杀招：保留幅值的防爆点积 =================
        # 绝对不使用 F.normalize！我们要保留特征长度代表的“异常剧烈程度”！

        # 1. 计算未归一化的纯点积（包含角度与幅值）
        pos_sim = torch.sum(query * positive_key, dim=-1)  # [B]

        query_exp = query.unsqueeze(1)
        neg_keys_exp = negative_keys.unsqueeze(0)
        neg_sim = torch.sum(query_exp * neg_keys_exp, dim=-1)  # [B, N_neg]

        temp = torch.clamp(self.temperature, min=0.01)
        pos_logits = pos_sim / temp
        neg_logits = neg_sim / temp

        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)

        # 🛡️ 物理防爆锁：限制 logits 的极值！
        # InfoNCE 内部计算 softmax 会用到 exp(logits)。
        # 当 logits 超过 88 时，float32 的 exp 就会溢出变 NaN。
        # 我们把它死死卡在 50，既保留了极其明显的正负样本差异，又绝对不会报错！
        logits = torch.clamp(logits, max=50.0)
        # =====================================================================

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
        return F.cross_entropy(logits, labels)