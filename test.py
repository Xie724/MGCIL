import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve


def smooth_scores(scores, sigma=2.0):
    """高斯时序平滑：消除帧级异常分数的孤立尖峰。

    暴力帧通常持续多帧（连贯），孤立的单帧高分往往是误判。
    用高斯核沿时间轴平滑，让分数序列更连贯。
    scores: [T] 帧级分数
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0).unsqueeze(0)          # [1, 1, T]
    kernel_size = int(sigma * 6) | 1                       # 奇数核
    x = torch.arange(kernel_size, dtype=torch.float32, device=scores.device) - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    gauss = gauss.view(1, 1, -1)
    smoothed = F.conv1d(scores, gauss, padding=kernel_size // 2)
    return smoothed.squeeze()


# 定义avce_test函数，用于对模型进行测试并计算相关指标
def avce_test(dataloader, model_av, model_v, gt, e, use_gts=True, sigma=2.0):
    """测试并返回 segment-level Average Precision (AP) of the av / v branches.

    Returns:
        av_ap (float): AP of the audio-visual fusion branch (XD-Violence metric).
        v_ap   (float): AP of the visual-only branch, or 0.0 if model_v is None.
    """
    with torch.no_grad():
        # 将模型设置为评估模式
        model_av.eval()
        pred = torch.zeros(0).cuda()
        if model_v is not None:
            model_v.eval()
            pred3 = torch.zeros(0).cuda()

        for i, (f_v, f_a) in enumerate(dataloader):
            f_v, f_a = f_v.cuda(), f_a.cuda()
            # 前向：忽略前三个输出，取 av_logits（融合分支）、audio_rep、visual_rep
            _, _, _, av_logits, audio_rep, visual_rep, _, _ = model_av(f_a, f_v, seq_len=None)
            av_logits = torch.squeeze(av_logits)
            av_logits = torch.sigmoid(av_logits)
            # 在第0维上求均值，处理多裁剪（5-crop）的情况
            av_logits = torch.mean(av_logits, 0)
            # 时序高斯平滑（GTS，可选）
            if use_gts:
                av_logits = smooth_scores(av_logits, sigma=sigma)
            pred = torch.cat((pred, av_logits))

            if model_v is not None:
                v_logits = model_v(f_v, seq_len=None)
                v_logits = torch.squeeze(v_logits)
                v_logits = torch.sigmoid(v_logits)
                v_logits = torch.mean(v_logits, 0)
                if use_gts:
                    v_logits = smooth_scores(v_logits, sigma=sigma)
                pred3 = torch.cat((pred3, v_logits))

        # 计算 av 分支 AP（precision-recall 曲线下面积，帧级标签按16倍重复对齐）
        pred = list(pred.cpu().detach().numpy())
        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        av_ap = auc(recall, precision)

        if model_v is not None:
            pred3 = list(pred3.cpu().detach().numpy())
            precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred3, 16))
            v_ap = auc(recall, precision)
            return av_ap, v_ap
        else:
            return av_ap, 0.0
