"""
fig2_pipeline.py -- concept draft of the MGCIL framework (architecture overview).

Schematic flow (no measured data): pre-extracted features -> MACIL-SD backbone
(fc + cross-attention) -> audio/visual scores and the video-level prediction z;
CGM builds confidence-gated pools; ANC hardens negatives; SAO anchors the pools to
frozen orthogonal CLIP text anchors; VLC contrasts whole videos; the total loss is
computed on the pools; GTS smooths the output scores; the visual-only teacher is
updated by EMA. Run on a machine with matplotlib:
    python fig2_pipeline.py --out figs/fig2_pipeline
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, text, fc='#eaf3fb', ec='#1f77b4', fs=8, lw=1.1, tc='black'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc)


def arrow(ax, x0, y0, x1, y1, color='#444444', lw=1.4, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=11, color=color, lw=lw, linestyle=ls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figs/fig2_pipeline')
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

    # data
    box(ax, 0.3, 4.5, 1.7, 1.0, 'Visual\n(I3D, $F_V$)', fc='#fde9d9', ec='#e67e22')
    box(ax, 0.3, 0.4, 1.7, 1.0, 'Audio\n(VGGish, $F_A$)', fc='#fde9d9', ec='#e67e22')
    # backbone
    box(ax, 2.7, 1.9, 2.1, 2.2, 'MACIL-SD backbone\n$fc$ + Cross-Attention\n(scores $p^a,p^v$,\nvideo-level $z$)', fs=8)
    arrow(ax, 2.0, 5.0, 2.7, 3.4)
    arrow(ax, 2.0, 0.9, 2.7, 2.4)

    # granularity modules (right column)
    box(ax, 5.7, 4.4, 2.2, 1.0, 'CGM: confidence-gated\npool mining', fc='#fff2cc', ec='#b58900')
    box(ax, 5.7, 3.1, 2.2, 1.0, 'ANC: adversarial\nnegatives', fc='#ffe0e0', ec='#d62728')
    box(ax, 5.7, 1.8, 2.2, 1.0, 'SAO: CLIP semantic\nanchors (orthogonal)', fc='#f3e5f5', ec='#7b1fa2')
    box(ax, 5.7, 0.5, 2.2, 1.0, 'VLC: video-level\ncontrast', fc='#e8f5e9', ec='#2ca02c')
    arrow(ax, 4.8, 3.2, 5.7, 4.4)
    arrow(ax, 4.8, 3.0, 5.7, 3.6)
    arrow(ax, 4.8, 2.8, 5.7, 2.3)
    arrow(ax, 4.8, 2.6, 5.7, 1.0)

    # losses & output
    box(ax, 8.5, 2.4, 1.35, 1.4, 'Total\nloss\n$\\mathcal{L}$', fc='#f0f0f0', ec='#555555')
    arrow(ax, 7.9, 4.9, 8.5, 3.7)
    arrow(ax, 7.9, 3.6, 8.5, 3.3)
    arrow(ax, 7.9, 2.3, 8.5, 2.9)
    arrow(ax, 7.9, 1.0, 8.5, 2.6)

    # output path + GTS + teacher
    box(ax, 2.7, 5.0, 2.2, 0.8, 'visual-only teacher\n(EMA, $\\mu$)', fc='#e0f7fa', ec='#0097a7', fs=7)
    arrow(ax, 4.9, 5.3, 8.5, 3.9, color='#0097a7', ls=':')
    box(ax, 8.5, 5.0, 1.35, 0.7, 'GTS\nsmoothing', fc='#ffe0b2', ec='#ef6c00', fs=7)
    arrow(ax, 4.6, 2.2, 8.5, 5.0, color='#ef6c00', ls=':')
    ax.text(8.6, 4.6, 'final\nscores', fontsize=7, ha='center')

    ax.set_title('MGCIL: Multi-Granular Contrastive Learning (schematic)', fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=200)
    print(f'wrote {args.out}.pdf/.png')


if __name__ == '__main__':
    main()
