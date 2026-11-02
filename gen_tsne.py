"""
gen_tsne.py -- Fig.4: t-SNE of test-set segment representations (normal vs violent).

Ground truth: by default from list/gt.npy sliced per video in test-list order (same
labels used for the AP metric). Optional per-video json (--gt_json) supported too.

Collects one representation per video snippet (= cross-attention outputs averaged
over the 5 crops and across modalities), labels each snippet violent if any of its
16 frames is positive, and runs t-SNE on a stratified sample.

Usage (server, from MGCIL_release):
    python gen_tsne.py --model_dir ckpt/full_s1888.pkl --out figs/tsne --n 2500
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import option
from avce_network import AVCE_Model as Model
from avce_dataset import Dataset


def video_key_from_path(path):
    name = os.path.splitext(os.path.basename(path).strip())[0]
    return name.rsplit('__', 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='ckpt/full_s1888.pkl')
    ap.add_argument('--gt_json', default='', help='optional per-video json; empty -> use list/gt.npy')
    ap.add_argument('--out', default='figs/tsne')
    ap.add_argument('--n', type=int, default=2500, help='max number of points to embed')
    ap.add_argument('--perplexity', type=int, default=30)
    args = ap.parse_args()

    base = option.parser.parse_args([])
    base.model_dir = args.model_dir

    label_dict = None
    if args.gt_json:
        import json
        with open(args.gt_json, encoding='utf-8') as f:
            label_dict = json.load(f)
        print(f'[gt] using per-video json: {args.gt_json}')
    else:
        print('[gt] using list/gt.npy sliced in test-list order')

    gt_all = np.load(base.gt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(base).to(device)
    model.load_state_dict({k.replace('module.', ''): v
                           for k, v in torch.load(args.model_dir, map_location=device).items()})
    model.eval()

    loader = DataLoader(Dataset(base, test_mode=True), batch_size=5, shuffle=False,
                        num_workers=0, pin_memory=True)

    reps, labels = [], []
    frame_offset = 0
    with torch.no_grad():
        for i, (f_v, f_a) in enumerate(loader):
            key = video_key_from_path(loader.dataset.list[i * 5])
            f_v, f_a = f_v.to(device), f_a.to(device)
            _, _, _, _, v_out, a_out, _, _ = model(f_a, f_v, seq_len=None)   # [5, T, D]
            v_out = torch.mean(v_out, 0)     # [T, D] over the 5 crops
            a_out = torch.mean(a_out, 0)
            rep = ((v_out + a_out) / 2.0).cpu().numpy()     # [T, D]
            T = rep.shape[0]

            if label_dict is not None:
                entry = label_dict.get(key)
                if entry is None:
                    continue
                lab = np.array(entry['labels'])
            else:
                lab = gt_all[frame_offset:frame_offset + T * 16]
                frame_offset += T * 16

            nseg = min(T, len(lab) // 16)
            for j in range(nseg):
                reps.append(rep[j])
                labels.append(1 if lab[j * 16:(j + 1) * 16].any() else 0)

    reps = np.array(reps)
    labels = np.array(labels)
    print(f'collected {len(reps)} snippets: violent={labels.sum()} normal={(1 - labels).sum()}')

    if len(reps) > args.n:
        idx_v = np.where(labels == 1)[0]
        idx_n = np.where(labels == 0)[0]
        rng = np.random.RandomState(0)
        take_v = rng.choice(idx_v, min(len(idx_v), args.n // 2), replace=False)
        take_n = rng.choice(idx_n, min(len(idx_n), args.n // 2), replace=False)
        sel = np.concatenate([take_v, take_n])
        reps, labels = reps[sel], labels[sel]

    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, perplexity=args.perplexity, init='pca',
               random_state=0).fit_transform(reps)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lab_, c, name_ in [(0, '#1f77b4', 'Normal'), (1, '#d62728', 'Violent')]:
        m = labels == lab_
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.6, c=c, label=name_)
    ax.legend(markerscale=3)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=300)
    print(f'[done] t-SNE of {len(reps)} points -> {args.out}.png/.pdf')


if __name__ == '__main__':
    main()
