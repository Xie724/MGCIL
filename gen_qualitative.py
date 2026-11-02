"""
gen_qualitative.py -- Fig.3: per-video anomaly-score curves vs ground truth.

Two passes:
  1. score the whole test set and compute GLOBAL percentile thresholds of the
     predicted scores (median / 75% / 95%), as in the authors' earlier MGS figures;
  2. draw each requested video curve with the ground-truth bands and three dashed
     threshold lines (Median=gold, 75%=orange, 95%=red) that are identical on every
     figure.

Ground truth: list/gt.npy sliced per video in test-list order (no json needed).

Usage (server, from MGCIL_release):
    python gen_qualitative.py --model_dir ckpt/full_s1888.pkl --out figs/qual --gts 1 \
        --names "" --max 0
"""

import argparse
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import option
from avce_network import AVCE_Model as Model
from avce_dataset import Dataset
from test import smooth_scores


def video_key_from_path(path):
    name = os.path.splitext(os.path.basename(path).strip())[0]
    return name.rsplit('__', 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='ckpt/full_s1888.pkl')
    ap.add_argument('--out', default='figs/qual')
    ap.add_argument('--names', default='', help='comma-separated substrings to keep (empty=all)')
    ap.add_argument('--gts', type=int, default=1, help='apply Gaussian temporal smoothing')
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--max', type=int, default=0, help='cap number of saved figures (0=all)')
    args = ap.parse_args()

    base = option.parser.parse_args([])
    base.model_dir = args.model_dir
    keep = [s.strip().lower() for s in args.names.split(',') if s.strip()]

    gt_all = np.load(base.gt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(base).to(device)
    model.load_state_dict({k.replace('module.', ''): v
                           for k, v in torch.load(args.model_dir, map_location=device).items()})
    model.eval()

    loader = DataLoader(Dataset(base, test_mode=True), batch_size=5, shuffle=False,
                        num_workers=0, pin_memory=True)
    os.makedirs(args.out, exist_ok=True)

    # ---- pass 1: score whole test set; collect per-video snippets ----
    per_video = []            # (key, snippet scores [T], video labels [frames])
    all_scores = []
    frame_offset = 0
    with torch.no_grad():
        for i, (f_v, f_a) in enumerate(loader):
            key = video_key_from_path(loader.dataset.list[i * 5])
            f_v, f_a = f_v.to(device), f_a.to(device)
            _, _, _, av_logits, _, _, _, _ = model(f_a, f_v, seq_len=None)
            av_logits = torch.squeeze(av_logits)
            av_logits = torch.sigmoid(av_logits)
            av_snippet = torch.mean(av_logits, 0)
            if args.gts:
                av_snippet = smooth_scores(av_snippet, sigma=args.sigma)
            sn = av_snippet.cpu().numpy()
            all_scores.append(sn)
            lab = gt_all[frame_offset:frame_offset + sn.shape[0] * 16]
            frame_offset += sn.shape[0] * 16
            per_video.append((key, sn, lab))

    # reference levels over ALL predicted scores (whole test set); the median (~0.02,
    # near zero because normal segments dominate) is replaced by a conventional 0.5 line
    concat = np.concatenate(all_scores)
    th_75, th_95 = np.percentile(concat, 75), np.percentile(concat, 95)
    th_dec = 0.5
    print(f'[thresholds over all test scores] median={np.median(concat):.3f} '
          f'75%={th_75:.3f}  95%={th_95:.3f}  (drawn lines: 0.5 / {th_75:.3f} / {th_95:.3f})')

    all_frames = np.repeat(concat, 16)
    mlen = min(len(all_frames), len(gt_all))
    from sklearn.metrics import average_precision_score
    print(f'[sanity] overall AP = {average_precision_score(gt_all[:mlen], all_frames[:mlen]):.4f} '
          f'(expect ~0.8858 if --gts 1 and the reproduced full model is used)')

    # ---- pass 2: draw figures with the global threshold lines ----
    saved = 0
    for key, sn, lab in per_video:
        if keep and not any(s in key.lower() for s in keep):
            continue
        if args.max and saved >= args.max:
            continue
        pred = np.repeat(sn, 16)
        mlen = min(len(pred), len(lab))
        x = np.arange(mlen)

        fig, ax = plt.subplots(figsize=(12, 3.4))
        ax.plot(x, pred[:mlen], color='#1f77b4', lw=1.2, label='MGCIL')
        ax.fill_between(x, 0, lab[:mlen], where=lab[:mlen] > 0.5,
                        color='#d62728', alpha=0.35, label='GT')
        # reference lines (identical on every figure): 0.5 decision + 75/95 percentiles
        ax.axhline(y=th_dec, color='#FFD700', ls='--', lw=1.0, alpha=0.9,
                   label=f'0.5')
        ax.axhline(y=th_75, color='#FF8C00', ls='--', lw=1.0, alpha=0.9,
                   label=f'75% ({th_75:.2f})')
        ax.axhline(y=th_95, color='#FF0000', ls='--', lw=1.0, alpha=0.9,
                   label=f'95% ({th_95:.2f})')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Frame'); ax.set_ylabel('Anomaly Score')
        ax.set_title(key, fontsize=8)
        ax.legend(loc='upper right', fontsize='small')
        fig.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9_.-]', '_', key)
        fig.savefig(os.path.join(args.out, f'{saved:04d}_{safe}.png'), dpi=200)
        plt.close(fig)
        saved += 1

    print(f'[done] saved {saved} figures to {args.out}')


if __name__ == '__main__':
    main()
