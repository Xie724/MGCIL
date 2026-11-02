"""
plot_sensitivity.py -- sensitivity plots, one chart type per parameter.

Chart types:
    tau          -> line plot (numeric axis)
    lamda_anchor -> bar chart (equal-spaced categories)
    lamda_video  -> filled area, equal-spaced x (avoids visual gaps for uneven sampling)
    conf_temp    -> stem plot (numeric axis)

The chosen (paper) value is highlighted and uses its measured AP when present in the
sweep (else the reproduced full-model AP 0.8858).

Usage:
    python plot_sensitivity.py --run runs/20260907-103002 --out figs/sensitivity
Outputs a combined 1x4 figure plus one separate figure per parameter.
"""

import argparse
import csv
import os
import re
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHOSEN = {
    'tau': 0.14,
    'lamda_anchor': 0.025,
    'lamda_video': 0.01,
    'conf_temp': 0.2,
}
CHOSEN_AP = 0.8858
STYLE = {
    'tau': 'line',
    'lamda_anchor': 'bar',
    'lamda_video': 'area',
    'conf_temp': 'stem',
}
CAT_KEYS = {'lamda_anchor', 'lamda_video'}   # equal-spaced (categorical) x-axis
KEYS = ['tau', 'lamda_anchor', 'lamda_video', 'conf_temp']
TITLES = {
    'tau': r'InfoNCE temperature $\tau$',
    'lamda_anchor': r'Semantic-anchor weight $\lambda_{\mathrm{sa}}$',
    'lamda_video': r'Video-level weight $\lambda_{\mathrm{vlc}}$',
    'conf_temp': r'Gating temperature $\tau_c$',
}
XLIM = {
    'tau': (0.06, 0.22),
    'conf_temp': (0.0, 0.65),
}
CB, COR = '#1f77b4', '#d62728'
Y0 = 0.84


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    sens = {k: [] for k in KEYS}
    chosen_measured = {k: CHOSEN_AP for k in KEYS}
    for r in rows:
        m = re.match(r'sens_(\w+)_([\d.]+)_s\d+', r['name'])
        if not m or m.group(1) not in sens:
            continue
        x, y = float(m.group(2)), float(r['best_av_ap'])
        sens[m.group(1)].append((x, y))
        if abs(x - CHOSEN[m.group(1)]) < 1e-9:
            chosen_measured[m.group(1)] = y
    return {k: sorted(v) for k, v in sens.items()}, chosen_measured


def draw(ax, key, pts, chosen_y):
    raw_x = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    style = STYLE[key]
    cat = key in CAT_KEYS                     # equal-spaced x (no visual holes)
    px = list(range(len(raw_x))) if cat else raw_x
    cx = CHOSEN[key]
    cxi = raw_x.index(cx) if cx in raw_x else None

    if style == 'line':
        ax.plot(px, ys, '-o', color=CB, ms=4, lw=1.5, zorder=2)
        ax.plot(cxi if cat else cx, chosen_y, marker='*', ms=15, color=COR, ls='none', zorder=3)
    elif style == 'bar':
        ax.bar(px, ys, width=0.65,
               color=[COR if i == cxi else CB for i in range(len(px))], zorder=2)
        ax.set_xlim(-0.6, len(px) - 0.4)
        for i, y in zip(px, ys):
            ax.annotate(f'{y:.3f}', (i, y), textcoords='offset points',
                        xytext=(0, 3), ha='center', fontsize=6)
    elif style == 'area':
        ax.plot(px, ys, '-', color=CB, lw=1.5, zorder=2)
        ax.fill_between(px, Y0, ys, color=CB, alpha=0.25, zorder=1)
        ax.plot(cxi if cat else cx, chosen_y, marker='*', ms=15, color=COR, ls='none', zorder=3)
    elif style == 'stem':
        for x, y in zip(px, ys):
            ax.plot([x, x], [Y0, y], color='#7f7f7f', lw=1.0, zorder=1)
            ax.plot(x, y, 'o', ms=4,
                    color=COR if x == cx else CB, zorder=2)

    if cat:
        ax.set_xticks(px)
        ax.set_xticklabels([f'{x:g}' for x in raw_x], fontsize=7, rotation=45, ha='right')
    else:
        ax.set_xlim(*XLIM[key])

    ax.set_title(TITLES[key], fontsize=9)
    ax.set_ylim(0.84, 0.90)
    ax.set_ylabel('AP')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)


def save(fig, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}', dpi=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None, help='run dir under runs/ (default: latest)')
    ap.add_argument('--out', default='sensitivity')
    args = ap.parse_args()

    run_dir = args.run or sorted(glob.glob(os.path.join('runs', '20*')))[-1]
    csv_path = os.path.join(run_dir, 'summary_results.csv')
    sens, chosen_y = load(csv_path)
    print(f'[data] {run_dir}')

    base = os.path.dirname(args.out) or '.'
    stem = os.path.basename(args.out)

    # two 2-panel figures (each placed at a different spot in the paper)
    for pi, pair in enumerate([('tau', 'lamda_anchor'), ('lamda_video', 'conf_temp')], start=1):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
        for ax, key in zip(axes, pair):
            draw(ax, key, sens[key], chosen_y[key])
        fig.tight_layout()
        save(fig, os.path.join(base, f'{stem}_part{pi}'))
        plt.close(fig)

    # one separate figure per parameter (optional)
    for key in KEYS:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        draw(ax, key, sens[key], chosen_y[key])
        fig.tight_layout()
        save(fig, os.path.join(base, f'sens_{key}'))
        plt.close(fig)

    print(f'wrote {stem}_part1/2.pdf|png (2 panels each) + 4 per-parameter figures')


if __name__ == '__main__':
    main()
