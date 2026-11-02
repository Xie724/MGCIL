import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from InfoNCE import InfoNCE


def adversarial_negatives(query, negatives, eps=0.005):
    """Generate adversarial negatives via FGSM toward the positive (query) direction.

    Core insight: the InfoNCE gradient w.r.t. a negative n_j is proportional to q
    (the query vector). Perturbing negatives toward q creates "worst-case" negatives
    that maximally confuse the contrastive loss, forcing the encoder to learn
    modality-invariant features that are robust to this perturbation.

    No memory, no prototypes, no extra parameters -- just one FGSM step per batch.
    """
    with torch.no_grad():
        attack = F.normalize(query.mean(dim=0, keepdim=True), dim=-1)     # [1, D]
        neg_adv = F.normalize(negatives + eps * attack, dim=-1)           # [N_neg, D]
    return neg_adv


def _pool_soft(mmil_logits, audio_logits, visual_logits, seq_len, audio_rep, visual_rep, conf_temp=0.2):
    """MGCIL confidence-gated soft semi-bag mining (CGM).

    Replaces the hard >0.5 binary assignment of MACIL-SD: each video contributes to
    the abnormal / normal pools with a multiplicity proportional to a sigmoid-gated
    confidence. Because w_nor = 1 - w_abn, videos are split into three regimes:
    w_abn > 0.65 -> abnormal pool only; w_abn < 0.35 -> normal pool only; and
    0.35 < w_abn < 0.65 -> BOTH pools, with small, nearly equal multiplicities that
    largely cancel, instead of being forcibly committed to one side.

    conf_temp (tau_c): lower = sharper decisions near the 0.5 boundary.
    """
    audio_abn = torch.zeros(0).cuda()
    visual_abn = torch.zeros(0).cuda()
    audio_bgd = torch.zeros(0).cuda()
    visual_bgd = torch.zeros(0).cuda()
    audio_nor = torch.zeros(0).cuda()
    visual_nor = torch.zeros(0).cuda()

    for i in range(audio_logits.shape[0]):
        logit = (mmil_logits[i].detach() - 0.5) / conf_temp
        w_abn = torch.sigmoid(logit)        # ~1.0 confident abnormal, ~0.0 confident normal
        w_nor = 1.0 - w_abn

        k = int(seq_len[i] // 16 + 1)

        _, a_topk_idx = torch.topk(audio_logits[i][:seq_len[i]], k=k, largest=True)
        _, v_topk_idx = torch.topk(visual_logits[i][:seq_len[i]], k=k, largest=True)
        a_topk_rep = audio_rep[i][a_topk_idx]       # [K, D]
        v_topk_rep = visual_rep[i][v_topk_idx]      # [K, D]

        _, a_botk_idx = torch.topk(audio_logits[i][:seq_len[i]], k=k, largest=False)
        _, v_botk_idx = torch.topk(visual_logits[i][:seq_len[i]], k=k, largest=False)
        a_botk_rep = audio_rep[i][a_botk_idx]
        v_botk_rep = visual_rep[i][v_botk_idx]

        # Abnormal pool: mean-expanded top-k (following original design)
        a_topk_mean = torch.mean(a_topk_rep, 0, keepdim=True).expand(k, -1)
        v_topk_mean = torch.mean(v_topk_rep, 0, keepdim=True).expand(k, -1)

        # Confidence-gated contribution. u = (z - 0.5) / tau_c, w_abn = sigmoid(u).
        # Since w_nor = 1 - w_abn, the two 0.35 gates partition videos into three
        # regimes: w_abn > 0.65 -> abnormal only; w_abn < 0.35 -> normal only;
        # 0.35 < w_abn < 0.65 -> BOTH pools (2-3 copies each, votes largely cancel).
        # A pool receives 0 copies only when its own gate (w > 0.35) is not passed;
        # note a confident-abnormal video has w_nor < 0.35 and hence 0 normal copies.
        n_abn_copies = int(w_abn * 5 + 0.5) if w_abn > 0.35 else 0   # 0 or 2-5 copies
        n_nor_copies = int(w_nor * 5 + 0.5) if w_nor > 0.35 else 0

        for _ in range(n_abn_copies):
            audio_abn = torch.cat((audio_abn, a_topk_mean), 0)
            visual_abn = torch.cat((visual_abn, v_topk_mean), 0)

        for _ in range(n_nor_copies):
            audio_nor = torch.cat((audio_nor, a_topk_rep), 0)
            visual_nor = torch.cat((visual_nor, v_topk_rep), 0)

        # Background pool: always contributes once
        audio_bgd = torch.cat((audio_bgd, a_botk_rep), 0)
        visual_bgd = torch.cat((visual_bgd, v_botk_rep), 0)

    return audio_abn, visual_abn, audio_bgd, visual_bgd, audio_nor, visual_nor


def _pool_hard(mmil_logits, audio_logits, visual_logits, seq_len, audio_rep, visual_rep):
    """Original MACIL-SD hard-threshold semi-bag pooling (used when use_cgm=0).

    A video with video-level score > 0.5 contributes its (mean-expanded) top-k to the
    abnormal pool; otherwise its raw top-k goes to the normal pool. Bottom-k always
    forms the background pool.
    """
    audio_abn = torch.zeros(0).cuda()
    visual_abn = torch.zeros(0).cuda()
    audio_bgd = torch.zeros(0).cuda()
    visual_bgd = torch.zeros(0).cuda()
    audio_nor = torch.zeros(0).cuda()
    visual_nor = torch.zeros(0).cuda()

    for i in range(audio_logits.shape[0]):
        k = int(seq_len[i] // 16 + 1)

        _, cur_visual_inv_idx = torch.topk(visual_logits[i][:seq_len[i]], k=k, largest=False)
        cur_visual_inv_rep = visual_rep[i][cur_visual_inv_idx]
        visual_bgd = torch.cat((visual_bgd, cur_visual_inv_rep), 0)

        _, cur_audio_inv_idx = torch.topk(audio_logits[i][:seq_len[i]], k=k, largest=False)
        cur_audio_inv_rep = audio_rep[i][cur_audio_inv_idx]
        audio_bgd = torch.cat((audio_bgd, cur_audio_inv_rep), 0)

        if mmil_logits[i] > 0.5:
            # abnormal video: mean-expanded top-k into abnormal pools
            _, cur_audio_idx = torch.topk(audio_logits[i][:seq_len[i]], k=k, largest=True)
            cur_audio_rep = audio_rep[i][cur_audio_idx]
            cur_dim = cur_audio_rep.size()
            cur_audio_rep = torch.mean(cur_audio_rep, 0, keepdim=True).expand(cur_dim)
            audio_abn = torch.cat((audio_abn, cur_audio_rep), 0)

            _, cur_visual_idx = torch.topk(visual_logits[i][:seq_len[i]], k=k, largest=True)
            cur_visual_rep = visual_rep[i][cur_visual_idx]
            cur_dim = cur_visual_rep.size()
            cur_visual_rep = torch.mean(cur_visual_rep, 0, keepdim=True).expand(cur_dim)
            visual_abn = torch.cat((visual_abn, cur_visual_rep), 0)
        else:
            # normal video: raw top-k into normal pools
            _, cur_audio_idx = torch.topk(audio_logits[i][:seq_len[i]], k=k, largest=True)
            cur_audio_rep = audio_rep[i][cur_audio_idx]
            audio_nor = torch.cat((audio_nor, cur_audio_rep), 0)

            _, cur_visual_idx = torch.topk(visual_logits[i][:seq_len[i]], k=k, largest=True)
            cur_visual_rep = visual_rep[i][cur_visual_idx]
            visual_nor = torch.cat((visual_nor, cur_visual_rep), 0)

    return audio_abn, visual_abn, audio_bgd, visual_bgd, audio_nor, visual_nor


def CMAL(mmil_logits, audio_logits, visual_logits, seq_len, audio_rep, visual_rep,
         anchor_abn, anchor_nor, tau=0.1, adv_eps=0.005,
         use_cgm=True, use_adv=True, use_anchor=True, conf_temp=0.2):
    """Cross-modal contrastive loss with MGCIL switches.

    Returns (loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n, loss_anchor).

    - use_cgm=True   -> confidence-gated soft mining;  False -> MACIL-SD hard pooling.
    - use_adv=True   -> add FGSM adversarial-negative regularizer (ANC).
    - use_anchor=True-> add CLIP semantic-anchor contrastive loss (SAO).
    - conf_temp      -> gating temperature tau_c of CGM.
    """
    if use_cgm:
        audio_abn, visual_abn, audio_bgd, visual_bgd, audio_nor, visual_nor = _pool_soft(
            mmil_logits, audio_logits, visual_logits, seq_len, audio_rep, visual_rep, conf_temp)
    else:
        audio_abn, visual_abn, audio_bgd, visual_bgd, audio_nor, visual_nor = _pool_hard(
            mmil_logits, audio_logits, visual_logits, seq_len, audio_rep, visual_rep)

    cmals = InfoNCE(negative_mode='unpaired', temperature=tau)

    if audio_nor.size(0) == 0 or audio_abn.size(0) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # --- a2v_a2b: audio_abn (query) vs visual_abn (positive), visual_bgd (neg) ---
    loss_a2v_a2b = cmals(audio_abn, visual_abn, visual_bgd)
    if use_adv:
        loss_a2v_a2b += 0.5 * cmals(audio_abn, visual_abn,
                                    adversarial_negatives(audio_abn, visual_bgd, adv_eps))

    # --- a2v_a2n: audio_abn (query) vs visual_abn (positive), visual_nor (neg) ---
    loss_a2v_a2n = cmals(audio_abn, visual_abn, visual_nor)
    if use_adv:
        loss_a2v_a2n += 0.5 * cmals(audio_abn, visual_abn,
                                    adversarial_negatives(audio_abn, visual_nor, adv_eps))

    # --- v2a_a2b: visual_abn (query) vs audio_abn (positive), audio_bgd (neg) ---
    loss_v2a_a2b = cmals(visual_abn, audio_abn, audio_bgd)
    if use_adv:
        loss_v2a_a2b += 0.5 * cmals(visual_abn, audio_abn,
                                    adversarial_negatives(visual_abn, audio_bgd, adv_eps))

    # --- v2a_a2n: visual_abn (query) vs audio_abn (positive), audio_nor (neg) ---
    loss_v2a_a2n = cmals(visual_abn, audio_abn, audio_nor)
    if use_adv:
        loss_v2a_a2n += 0.5 * cmals(visual_abn, audio_abn,
                                    adversarial_negatives(visual_abn, audio_nor, adv_eps))

    # --- SAO anchor loss (only when use_anchor) ---
    loss_anchor = 0.0
    if use_anchor:
        batch_abn = audio_abn.size(0)
        anc_abn_exp = anchor_abn.expand(batch_abn, -1)
        loss_anchor = cmals(visual_abn, anc_abn_exp, anchor_nor) + cmals(audio_abn, anc_abn_exp, anchor_nor)

        batch_nor = audio_nor.size(0)
        anc_nor_exp = anchor_nor.expand(batch_nor, -1)
        loss_anchor = loss_anchor + cmals(visual_nor, anc_nor_exp, anchor_abn) + cmals(audio_nor, anc_nor_exp, anchor_abn)

    return loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n, loss_anchor
