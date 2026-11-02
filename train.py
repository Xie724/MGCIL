import torch
# 从CMA_MIL模块中导入CMAL函数，该函数可能用于计算某种特定的损失
from CMA_MIL import CMAL
from InfoNCE import InfoNCE
import math


def video_level_contrastive(audio_rep, visual_rep, mmil_logits, seq_len, tau=0.1):
    """视频级对比损失：让异常视频整体相似、正常视频整体相似、两类互斥。

    补充帧级对比没有的"视频整体语义"。每段视频取其帧表征均值作为视频表征，
    异常视频表征互相靠近、远离正常视频表征，反之亦然。

    audio_rep, visual_rep: [B, T, D]
    mmil_logits: [B] 视频级预测
    seq_len: [B]
    """
    B = audio_rep.shape[0]
    video_reps = []
    for i in range(B):
        sl = seq_len[i]
        a_mean = audio_rep[i][:sl].mean(dim=0)
        v_mean = visual_rep[i][:sl].mean(dim=0)
        video_reps.append((a_mean + v_mean) / 2)
    video_reps = torch.stack(video_reps)                       # [B, D]

    abn_idx = mmil_logits > 0.5
    nor_idx = mmil_logits <= 0.5

    if abn_idx.sum() < 2 or nor_idx.sum() < 2:
        return torch.tensor(0.0, device=audio_rep.device)

    abn_reps = video_reps[abn_idx]                             # [N_abn, D]
    nor_reps = video_reps[nor_idx]                             # [N_nor, D]

    cmals = InfoNCE(negative_mode='unpaired', temperature=tau)

    # 异常视频：以异常均值作正样本，正常视频作负样本
    abn_mean = abn_reps.mean(dim=0, keepdim=True).expand_as(abn_reps)
    loss_abn = cmals(abn_reps, abn_mean, nor_reps)

    # 正常视频：以正常均值作正样本，异常视频作负样本
    nor_mean = nor_reps.mean(dim=0, keepdim=True).expand_as(nor_reps)
    loss_nor = cmals(nor_reps, nor_mean, abn_reps)

    return loss_abn + loss_nor


def uncertainty_consistency_loss(audio_scores, visual_scores, seq_len, eps=1e-6):
    """基于模态不确定性的跨模态一致性约束损失。

    audio_scores: [B, T]
    visual_scores: [B, T]
    seq_len: [B]
    """
    if audio_scores.dim() != 2 or visual_scores.dim() != 2:
        audio_scores = audio_scores.view(audio_scores.size(0), -1)
        visual_scores = visual_scores.view(visual_scores.size(0), -1)

    # 防止 log(0) / log(1) 数值不稳定
    a = audio_scores.clamp(eps, 1.0 - eps)
    v = visual_scores.clamp(eps, 1.0 - eps)
    u_a = -(a * torch.log(a) + (1.0 - a) * torch.log(1.0 - a))
    u_v = -(v * torch.log(v) + (1.0 - v) * torch.log(1.0 - v))
    diff2 = (v - a) ** 2
    weighted = (torch.exp(-u_v) + torch.exp(-u_a)) * diff2

    max_len = audio_scores.size(1)
    seq_len = seq_len.to(audio_scores.device)
    mask = torch.arange(max_len, device=audio_scores.device).unsqueeze(0) < seq_len.unsqueeze(1)
    valid = weighted.masked_select(mask)
    if valid.numel() == 0:
        return torch.tensor(0.0, device=audio_scores.device)
    return valid.mean()


# 定义avce_train函数，用于训练音频 - 视觉融合模型（model_av）和视觉模型（model_v）
# 参数说明：
# dataloader: 数据加载器，用于加载训练数据
# model_av: 音频 - 视觉融合模型
# model_v: 视觉模型
# optimizer_av: 音频 - 视觉融合模型的优化器
# optimizer_v: 视觉模型的优化器
# criterion: 损失函数，用于计算损失
# lamda_a2b: 超参数，用于加权某些损失项
# lamda_a2n: 超参数，用于加权某些损失项
# logger: 日志记录器，用于记录训练过程中的信息
def avce_train(dataloader, model_av, model_v, optimizer_av, optimizer_v, criterion, lamda_a2b, lamda_a2n, lamda_anchor, tau, logger, adv_eps=0.005, lamda_video=0.1, use_cgm=True, use_uwc=True, use_anchor=True, use_adv=True, use_vlc=True, grad_clip=1.0, conf_temp=0.2):
    # 启用梯度计算，确保在训练过程中可以进行反向传播
    with torch.set_grad_enabled(True):
        # 将音频 - 视觉融合模型设置为训练模式，开启一些在训练时需要的特殊层（如Dropout和BatchNorm）
        model_av.train()
        # 将视觉模型设置为训练模式
        model_v.train()
        # 遍历数据加载器中的每个批次
        for i, (f_v, f_a, label) in enumerate(dataloader):
            # 计算每个样本的有效序列长度
            # 首先计算f_v在维度2上的绝对值的最大值，然后判断哪些值大于0，最后在维度1上求和得到每个样本的有效长度
            seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
            # 根据计算得到的有效序列长度，截取f_v的有效部分
            f_v = f_v[:, :torch.max(seq_len), :]
            # 根据计算得到的有效序列长度，截取f_a的有效部分
            f_a = f_a[:, :torch.max(seq_len), :]
            # 将视觉特征f_v、音频特征f_a和标签label转换为浮点数类型，并移动到GPU上
            f_v, f_a, label = f_v.float().cuda(), f_a.float().cuda(), label.float().cuda()
            # 调用音频 - 视觉融合模型进行前向传播，获取输出结果
            # mmil_logits: 某种多实例学习的逻辑值
            # audio_logits: 音频的逻辑值
            # visual_logits: 视觉的逻辑值
            # 忽略第四个输出
            # audio_rep: 音频表示
            # visual_rep: 视觉表示
            mmil_logits, audio_logits, visual_logits, _, audio_rep, visual_rep, anchor_abn, anchor_nor = model_av(f_a, f_v, seq_len)
            # 去除audio_logits中维度为1的维度
            audio_logits = audio_logits.squeeze()
            # 去除visual_logits中维度为1的维度
            visual_logits = visual_logits.squeeze()
            # 去除mmil_logits中维度为1的维度
            mmil_logits = mmil_logits.squeeze()
            # 计算分类损失，使用传入的损失函数criterion
            clsloss = criterion(mmil_logits, label)
            # 调用CMAL函数计算不同的CMA损失
            # cmaloss_a2v_a2b, cmaloss_a2v_a2n, cmaloss_v2a_a2b, cmaloss_v2a_a2n = CMAL(mmil_logits, audio_logits,
            #                                                                           visual_logits, seq_len, audio_rep,
            #                                                                           visual_rep)
            # cmaloss_a2v_a2b, cmaloss_a2v_a2n, cmaloss_v2a_a2b, cmaloss_v2a_a2n, loss_anchor = CMAL(
            #     mmil_logits, audio_logits, visual_logits, seq_len, audio_rep, visual_rep, anchor_abn, anchor_nor
            # )
            cmaloss_a2v_a2b, cmaloss_a2v_a2n, cmaloss_v2a_a2b, cmaloss_v2a_a2n, loss_anchor = CMAL(
                mmil_logits, audio_logits, visual_logits, seq_len, audio_rep, visual_rep, anchor_abn, anchor_nor,
                tau, adv_eps, use_cgm, use_adv, use_anchor, conf_temp
            )
            # 不确定性加权一致性损失（仅 use_uwc 时计入）
            consistency_loss = uncertainty_consistency_loss(audio_logits, visual_logits, seq_len) if use_uwc \
                else torch.tensor(0.0, device=audio_logits.device)
            # 视频级对比损失：异常/正常视频整体语义分离（仅 use_vlc 时计入）
            video_loss = video_level_contrastive(audio_rep, visual_rep, mmil_logits, seq_len, tau) if use_vlc \
                else torch.tensor(0.0, device=audio_rep.device)
            # 计算总的损失，将分类损失、CMA损失和各项损失加权求和
            total_loss = clsloss + lamda_a2b * cmaloss_a2v_a2b + lamda_a2b * cmaloss_v2a_a2b + \
                         lamda_a2n * cmaloss_a2v_a2n + lamda_a2n * cmaloss_v2a_a2n + \
                         lamda_anchor * loss_anchor + consistency_loss + lamda_video * video_loss
            # 计算数据加载器长度的一半，用于记录日志的间隔
            unit = dataloader.__len__() // 2
            # 每隔unit个批次记录一次日志
            if i % unit == 0:
                # 记录当前的超参数lambda_a2b和lambda_a2n的值
                logger.info(f"Current Lambda_a2b: {lamda_a2b:.2f}, Current Lambda_a2n: {lamda_a2n:.2f}")
                # 记录当前的各种损失值
                logger.info(
                    f"{int(i // unit)}/{2} MIL Loss: {clsloss:.4f}, CMA Loss A2V_A2B: {cmaloss_a2v_a2b:.4f}, CMA Loss A2V_A2N: {cmaloss_a2v_a2n:.4f},"
                    f" CMA Loss V2A_A2B: {cmaloss_v2a_a2b:.4f}, CMA Loss V2A_A2N: {cmaloss_v2a_a2n:.4f}, "
                    f"Consistency Loss: {consistency_loss:.4f}, Video Loss: {video_loss:.4f}")

            # 调用视觉模型进行前向传播，获取输出结果
            v_logits = model_v(f_v, seq_len)
            # 计算视觉模型的损失
            loss_v = criterion(v_logits, label)

            # 清空音频 - 视觉融合模型的梯度
            optimizer_av.zero_grad()
            # 清空视觉模型的梯度
            optimizer_v.zero_grad()
            # 允许音频 - 视觉融合模型计算梯度
            model_av.requires_grad = True
            # 禁止视觉模型计算梯度
            model_v.requires_grad = False
            # 对总的损失进行反向传播，计算梯度
            total_loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_av.parameters(), max_norm=grad_clip)
            # 更新音频 - 视觉融合模型的参数
            optimizer_av.step()

            # 再次清空音频 - 视觉融合模型的梯度
            optimizer_av.zero_grad()
            # 再次清空视觉模型的梯度
            optimizer_v.zero_grad()
            # 禁止音频 - 视觉融合模型计算梯度
            model_av.requires_grad = False
            # 允许视觉模型计算梯度
            model_v.requires_grad = True
            # 对视觉模型的损失进行反向传播，计算梯度
            loss_v.backward()
            # 更新视觉模型的参数
            optimizer_v.step()

        # 返回总的损失和视觉模型的损失
        return total_loss, loss_v