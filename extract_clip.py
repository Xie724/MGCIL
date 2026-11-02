import torch
import clip
import torch.nn as nn

# 1. 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. 构建涵盖多种情况的 Prompt 池
abn_prompts = [
    "A video of a person shooting a gun.",
    "A video of a violent explosion and fire.",
    "A video of a vicious physical fight, martial arts, or brawl.",
    "A video of a riot, mob violence, or crowd panic.",
    "A video of a severe car crash or traffic accident.",
    "A video of physical abuse or assault."
]
nor_prompts = [
    "A video of people walking peacefully on the street.",
    "A video of normal daily activities and talking.",
    "A video of a calm and quiet background view.",
    "A video of surveillance footage with nothing unusual happening."
]

with torch.no_grad():
    # 提取特征并求平均 (Prompt Ensembling)
    abn_tokens = clip.tokenize(abn_prompts).to(device)
    nor_tokens = clip.tokenize(nor_prompts).to(device)

    feature_abn = model.encode_text(abn_tokens).float().mean(dim=0, keepdim=True)
    feature_nor = model.encode_text(nor_tokens).float().mean(dim=0, keepdim=True)

    # L2 归一化
    feature_abn = feature_abn / feature_abn.norm(dim=-1, keepdim=True)
    feature_nor = feature_nor / feature_nor.norm(dim=-1, keepdim=True)

# 3. 降维到 128 维以匹配你的模型 (使用固定正交矩阵)
proj = nn.Linear(512, 128, bias=False).to(device)
nn.init.orthogonal_(proj.weight)

with torch.no_grad():
    anchor_abn_128 = proj(feature_abn)
    anchor_nor_128 = proj(feature_nor)
    anchor_abn_128 = anchor_abn_128 / anchor_abn_128.norm(dim=-1, keepdim=True)
    anchor_nor_128 = anchor_nor_128 / anchor_nor_128.norm(dim=-1, keepdim=True)

# 4. 保存到本地
torch.save(anchor_abn_128.cpu(), 'anchor_abn_clip.pt')
torch.save(anchor_nor_128.cpu(), 'anchor_nor_clip.pt')
print("✅ CLIP 细粒度文本特征已成功提取并保存！")