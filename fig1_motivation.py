"""
fig1_motivation.py -- concept draft of the three weaknesses MGCIL addresses.

Panel (a): hard-threshold semi-bag assignment (boundary video contaminates a pool)
           vs. confidence-gated assignment (uncertain video diluted across both pools).
Panel (b): randomly sampled (easy) negatives far from the query vs. adversarially
           constructed (worst-case) negatives near the query.
Panel (c): segment-level supervision only vs. also video-level contrast.

Concept schematic (no measured data). Run on a machine with matplotlib:
    python fig1_motivation.py --out figs/fig1_motivation
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

Y0 = 0.0


def box(ax, x, y, w, h, text, fc='#eaf3fb', ec='#1f77b4', fs=8, lw=1.0, tc='black'):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc)


def arrow(ax, x0, y0, x1, y1, color='#333333', lw=1.4, ls='-'):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10,
                        color=color, lw=lw, linestyle=ls)
    ax.add_patch(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figs/fig1_motivation')
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # ---- (a) hard vs. soft assignment ----
    ax = axes[0]
    ax.text(0.02, 0.97, '(a) semi-bag assignment', fontsize=9, fontweight='bold')
    box(ax, 0.03, 0.62, 0.44, 0.26, 'Hard threshold\n$z_i>0.5$: boundary video\ninjected into a pool', fs=7)
    box(ax, 0.53, 0.62, 0.44, 0.26, 'CGM soft gate\nuncertain video enters\nBOTH pools (diluted)', fs=7, fc='#fef3e2', ec='#d62728')
    # a small "0.5" divider
    ax.plot([0.03, 0.97], [0.5, 0.5], color='#999999', lw=0.8, ls='--')
    ax.text(0.97, 0.505, '0.5', fontsize=6, ha='right', va='bottom')
    for xx, val, col in [(0.12, 0.28, '#1f77b4'), (0.24, 0.5, '#7f7f7f'), (0.36, 0.5, '#7f7f7f'), (0.48, 0.72, '#d62728')]:
        ax.plot([xx, xx], [0.12, 0.5], color=col, lw=2.2)
    ax.text(0.5, 0.30, 'video-level score $z_i$', fontsize=7, ha='center')

    # ---- (b) easy vs adversarial negatives ----
    ax = axes[1]
    ax.text(0.02, 0.97, '(b) negatives', fontsize=9, fontweight='bold')
    ax.scatter([0.18], [0.78], marker='o', s=90, color='#d62728', label='query/positive')
    ax.scatter([0.62, 0.78, 0.5], [0.8, 0.55, 0.62], marker='x', s=60, color='#1f77b4')  # easy neg far
    ax.scatter([0.30, 0.34, 0.22], [0.66, 0.6, 0.72], marker='x', s=90, color='#2ca02c')  # adversarial near
    for n in [(0.30, 0.66), (0.34, 0.6), (0.22, 0.72)]:
        arrow(ax, 0.18, 0.78, n[0], n[1], color='#2ca02c', lw=1.0, ls=':')
    ax.text(0.5, 0.34, 'random negatives\n(far, weak gradient)', fontsize=7, ha='center')
    ax.text(0.5, 0.14, 'ANC: FGSM-step negatives\n(near the query, hard)', fontsize=7, ha='center', color='#2ca02c')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ---- (c) granularities ----
    ax = axes[2]
    ax.text(0.02, 0.97, '(c) supervision granularity', fontsize=9, fontweight='bold')
    box(ax, 0.08, 0.62, 0.36, 0.26, 'segment-level contrast\n(CMAL / UWC)', fs=7)
    box(ax, 0.56, 0.62, 0.36, 0.26, 'video-level contrast\n(VLC: abnormal vs.\nnormal videos)', fs=7, fc='#e8f5e9', ec='#2ca02c')
    box(ax, 0.08, 0.16, 0.36, 0.26, 'semantic anchors\n(SAO: CLIP text)', fs=7, fc='#f3e5f5', ec='#7b1fa2')
    arrow(ax, 0.44, 0.75, 0.56, 0.75)
    arrow(ax, 0.26, 0.62, 0.26, 0.42)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=200)
    print(f'wrote {args.out}.pdf/.png')


if __name__ == '__main__':
    main()
