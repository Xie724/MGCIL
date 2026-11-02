"""
report_results.py -- turn a runner summary_results.csv into paper-ready numbers.

Usage:
    python report_results.py                      # use the latest runs/<timestamp>
    python report_results.py --run runs/20260905-214043

Output:
    * baseline seeds + mean +/- std
    * full (MGCIL) seeds + mean +/- std
    * ablation rows (in Table-3 order, from 'alloff' to 'vlc')
    * probe rows (retuning headroom)
    * LaTeX-ready lines you can paste into cas-sc-template.tex

Only real measured numbers are printed (no placeholders, no averaging over FAILED).
"""

import argparse
import csv
import glob
import os
import statistics as st


def latest_run_dir():
    runs = sorted(glob.glob(os.path.join('runs', '20*')))
    if not runs:
        raise SystemExit('no runs/ directory found')
    return runs[-1]


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(values):
    vals = [v for v in values if v is not None]
    if len(vals) == 0:
        return 'n/a (all failed)'
    if len(vals) == 1:
        return f'{vals[0]:.4f}'
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f'{m:.4f} ± {sd:.4f}  (n={len(vals)})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None)
    args = ap.parse_args()
    run_dir = args.run or latest_run_dir()
    csv_path = os.path.join(run_dir, 'summary_results.csv')
    if not os.path.exists(csv_path):
        raise SystemExit(f'not found: {csv_path}')
    rows = load(csv_path)

    base, full, ab, probe = [], [], [], []
    ab_order = ['ab_alloff', 'ab_cgm', 'ab_uwc', 'ab_sao', 'ab_anc', 'ab_vlc']
    ab_map = {}
    for r in rows:
        name = r['name']
        v = num(r.get('best_av_ap'))
        if name.startswith('base_orig_s'):
            base.append((name, v))
        elif name.startswith('full_s'):
            full.append((name, v))
        elif name.startswith('ab_'):
            ab_map[name] = v
        elif name.startswith('p_'):
            probe.append((name, v))

    print(f'RUN: {run_dir}')
    print('=' * 70)
    print('\n[Baseline (pure MACIL-SD, orig hyper, no GTS)]')
    for n, v in base:
        print(f'  {n:22s} {v:.4f}' if v is not None else f'  {n:22s} FAILED')
    print('  mean ± std :', summarize([v for _, v in base]))

    print('\n[Full MGCIL]')
    for n, v in full:
        print(f'  {n:22s} {v:.4f}' if v is not None else f'  {n:22s} FAILED')
    print('  mean ± std :', summarize([v for _, v in full]))

    print('\n[Ablation (Table 3 order, seed 1888)]')
    for n in ab_order:
        v = ab_map.get(n)
        print(f'  {n:12s} {v:.4f}' if v is not None else f'  {n:12s} FAILED')

    print('\n[Probe (retuning headroom)]')
    for n, v in probe:
        print(f'  {n:12s} {v:.4f}' if v is not None else f'  {n:12s} FAILED')

    print('\n' + '=' * 70)
    print('LaTeX-ready lines:')
    if full and len([v for _, v in full if v is not None]) >= 2:
        vals = [v for _, v in full if v is not None]
        print(f'  Full mean±std: {st.mean(vals):.4f} $\\pm$ {st.stdev(vals):.4f}')
    for n in ab_order:
        v = ab_map.get(n)
        if v is not None:
            print(f'  Table-3 row {n:10s} AP={v:.4f}')
    print('\nNOTE: pick which regime (A/B) a run used before trusting its numbers.')


if __name__ == '__main__':
    main()
