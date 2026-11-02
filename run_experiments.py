"""
run_experiments.py -- batch runner for MGCIL paper experiments (ablation / seeds / sensitivity)

Run from the MGCIL_release folder (same dir as main.py), e.g.:

    nohup python run_experiments.py --groups baseline        > log_baseline.log 2>&1 &   # 纯MACIL-SD复现(原超参)
    nohup python run_experiments.py --groups full            > log_full.log     2>&1 &   # MGCIL多种子
    nohup python run_experiments.py --groups gts             > log_gts.log      2>&1 &   # GTS开/关
    nohup python run_experiments.py --groups ablation        > log_ablation.log 2>&1 &   # 消融 Table3 逐行
    nohup python run_experiments.py --groups sensitivity     > log_sens.log     2>&1 &   # 敏感性

Options:
    --groups   comma-separated group names (default: all)
    --gpus     list of GPU ids to use (default: ["0","1","1","0"])
    --parallel number of concurrent processes (default: len(gpus))
    --only     substring filter on experiment name (dry useful)
    --dry      print the commands and exit without running

Output: runs/<timestamp>/<name>/training.log  and  runs/<timestamp>/summary_results.csv

NOTE: adapt the seed lists / sweep values at the top of this file to your needs.
"""

import os
import sys
import csv
import re
import argparse
import datetime
import json
import subprocess
from multiprocessing import Pool, Manager

# =============================================================================
# 0. 官方配置 (paper config = XS90 best trial_021) -- all component switches default ON.
# =============================================================================
OFFICIAL = {
    'lr': '0.0006', 'dropout': '0.4', 'tau': '0.14',
    'lamda_a2b': '0.7', 'lamda_a2n': '0.07', 'lamda_cof': '0.06',
    'lamda_anchor': '0.025', 'adv_eps': '0.002', 'lamda_video': '0.01',
    'm': '0.87',  # EMA teacher momentum (trial_021 value; option default is 0.91!)
    'batch-size': '128', 'max-epoch': '50', 'workers': '0',
    'hid_dim': '128', 'ffn_dim': '128', 'num_stages': '3', 'nhead': '4',
    'conf_temp': '0.2', 'gts_sigma': '2.0',
}
# Component switches: default ON (full MGCIL). Turn off for ablations / baseline.
SWITCHES = {
    'use_cgm': '1', 'use_uwc': '1', 'use_anchor': '1',
    'use_adv': '1', 'use_vlc': '1', 'use_gts': '1',
    'grad_clip': '1.0',
}
# Original MACIL-SD hyper-parameters (used to reproduce the 82.90 baseline).
ORIGINAL = {
    'lr': '0.0004', 'dropout': '0.1', 'tau': '0.1',
    'lamda_a2b': '1.5', 'lamda_a2n': '1.5', 'lamda_cof': '0.1',
    'm': '0.91',
    'batch-size': '128', 'max-epoch': '50', 'workers': '0',
    'hid_dim': '128', 'ffn_dim': '128', 'num_stages': '3', 'nhead': '4',
}

FIXED = ['--modality', 'MIX2',
         '--rgb-list', 'list/rgb.list', '--audio-list', 'list/audio.list',
         '--test-rgb-list', 'list/rgb_test.list', '--test-audio-list', 'list/audio_test.list',
         '--gt', 'list/gt.npy']

SEEDS_BASELINE = [1888, 2024, 42]
SEEDS_FULL = [1888, 2024, 42, 7, 123]   # 5 seeds -> robust mean +/- std

# =============================================================================
# 1. Experiment builders (each returns dict {name, params, switches})
# =============================================================================
def _make(name, params, switches, seed):
    return {'name': name, 'params': dict(params), 'switches': dict(switches), 'seed': seed}

def _switches_off():
    return {k: '0' for k in SWITCHES}

def build_baseline_orig(seed):
    """Pure MACIL-SD reproduction with the ORIGINAL MACIL-SD hyper-parameters."""
    sw = _switches_off()
    return _make(f'base_orig_s{seed}', dict(ORIGINAL), sw, seed)

def build_baseline_ours(seed):
    """Components off but under the OFFICIAL (paper) hyper-parameters."""
    sw = _switches_off()
    return _make(f'base_paper_s{seed}', dict(OFFICIAL), sw, seed)

def build_full(seed):
    return _make(f'full_s{seed}', dict(OFFICIAL), dict(SWITCHES), seed)

def build_gts(seed=1888):
    """GTS on/off for baseline and full, to quantify the post-processing gain."""
    sw_b = _switches_off()
    sw_f = dict(SWITCHES)
    return [
        _make(f'base_orig_gts0_s{seed}', dict(ORIGINAL), sw_b, seed),
        _make(f'base_orig_gts1_s{seed}', dict(ORIGINAL), dict(sw_b, use_gts='1'), seed),
        _make(f'full_gts0_s{seed}', dict(OFFICIAL), dict(sw_f, use_gts='0'), seed),
        _make(f'full_gts1_s{seed}', dict(OFFICIAL), dict(sw_f, use_gts='1'), seed),
    ]

def build_ablation(seed=1888):
    """Additive ablation (Table 3). All rows share the same evaluation
    (use_gts=1, grad_clip=1) so deltas isolate each training component.
    'alloff' = components off under official hyper-params; then add CGM, UWC, SAO, ANC, VLC."""
    sw = _switches_off()          # everything off but keep gts/clip on for identical eval
    sw = dict(sw, use_gts='1', grad_clip='1.0')
    # additive increments: each row keeps all components enabled by the previous rows
    order = [('use_cgm', 'ab_cgm'), ('use_uwc', 'ab_uwc'), ('use_anchor', 'ab_sao'),
             ('use_adv', 'ab_anc'), ('use_vlc', 'ab_vlc')]
    exps = [_make('ab_alloff', dict(OFFICIAL), dict(sw), seed)]
    enabled = dict(sw)
    for key, name in order:
        enabled[key] = '1'
        exps.append(_make(name, dict(OFFICIAL), dict(enabled), seed))
    return exps

def build_probe(seed=1888):
    """Quick retuning-headroom probes: full MGCIL with one hyper changed.
    (Robust regime; informs whether tuning can lift the multi-seed mean.)"""
    probes = [
        ('p_m91',      {'m': '0.91'}),
        ('p_tau11',    {'tau': '0.11'}),
        ('p_tau10',    {'tau': '0.10'}),
        ('p_lr4e4',    {'lr': '0.0004'}),
        ('p_adv1e3',   {'adv_eps': '0.001'}),
        ('p_anch05',   {'lamda_anchor': '0.05'}),
    ]
    return [_make(n, dict(OFFICIAL, **ov), dict(SWITCHES), seed) for n, ov in probes]


def build_sensitivity(seed=1888):
    """One-at-a-time sensitivity around the official config (full MGCIL, gts on).
    Denser sweeps (8-10 points each) so the plotted curves are smooth/long."""
    sweeps = {
        'tau':          ['0.08', '0.10', '0.12', '0.13', '0.14',
                         '0.15', '0.16', '0.18', '0.20'],
        'lamda_anchor': ['0.000', '0.005', '0.010', '0.015', '0.020',
                         '0.025', '0.030', '0.050'],
        'lamda_video':  ['0.000', '0.002', '0.005', '0.007', '0.010',
                         '0.015', '0.020', '0.050'],
        'conf_temp':    ['0.05', '0.10', '0.15', '0.20', '0.25',
                         '0.30', '0.40', '0.60'],
    }
    exps = []
    for key, vals in sweeps.items():
        for v in vals:
            params = dict(OFFICIAL, **{key: v})
            exps.append(_make(f'sens_{key}_{v}_s{seed}', params, dict(SWITCHES), seed))
    return exps

GROUPS = {
    'baseline':   [build_baseline_orig(s) for s in SEEDS_BASELINE],
    'baseline_ours': [build_baseline_ours(s) for s in SEEDS_BASELINE],
    'full':       [build_full(s) for s in SEEDS_FULL],
    'gts':        build_gts(),
    'ablation':   build_ablation(),
    'probe':      build_probe(),
    'sensitivity': build_sensitivity(),
}

# =============================================================================
# 2. Command builder + worker
# =============================================================================
def build_command(exp, gpu_id):
    cmd = [sys.executable, 'main.py']
    cmd += FIXED
    for k, v in exp['params'].items():
        cmd += [f'--{k}', str(v)]
    for k, v in exp['switches'].items():
        cmd += [f'--{k}', str(v)]
    cmd += ['--seed', str(exp['seed']), '--gpus', str(gpu_id), '--model-name', exp['name']]
    return cmd

def run_one(bundle):
    exp, run_dir, gpu_queue, csv_path, csv_lock, dry = bundle
    gpu_id = gpu_queue.get()
    name = exp['name']
    trial_dir = os.path.join(run_dir, name)
    os.makedirs(trial_dir, exist_ok=True)
    log_path = os.path.join(trial_dir, 'training.log')
    cmd = build_command(exp, gpu_id)

    if dry:
        print('DRY>', ' '.join(cmd))
        gpu_queue.put(gpu_id)
        return

    result = [name, gpu_id, 'FAILED', 'FAILED', 'FAILED', log_path]
    try:
        with open(log_path, 'w') as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
            proc.wait()
        if proc.returncode != 0:
            print(f"!!!!!! {name} on GPU {gpu_id} FAILED (rc={proc.returncode}). See {log_path}")
        else:
            with open(log_path) as lf:
                text = lf.read()
            m = re.search(r"Best Performance in Epoch (\d+):\s*av_ap:([\d.]+)\s*\|\s*v_ap:([\d.]+)", text)
            if m:
                result = [name, gpu_id, f"{float(m.group(2)):.4f}", f"{float(m.group(3)):.4f}", m.group(1), log_path]
            else:
                print(f"!!!!!! {name} finished but result line not parsed. See {log_path}")
    except Exception as e:
        print(f"!!!!!! {name} on GPU {gpu_id} EXCEPTION: {e}")
    finally:
        gpu_queue.put(gpu_id)
        with csv_lock:
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(result)
        print(f"--- finished {name} (GPU {gpu_id}) -> ap={result[2]} ---")
        return result

# =============================================================================
# 3. Main
# =============================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=None, help='comma-separated group names, default all')
    ap.add_argument('--gpus', nargs='+', default=['0', '1', '1', '0'], help='GPU ids (repeat to run multiple per gpu)')
    ap.add_argument('--parallel', type=int, default=None)
    ap.add_argument('--only', default=None, help='run only experiments whose name contains this substring')
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(',')] if args.groups else list(GROUPS.keys())
    exps = []
    for g in groups:
        if g not in GROUPS:
            print(f'[warn] unknown group {g}; available: {list(GROUPS.keys())}')
            continue
        exps += GROUPS[g]
    if args.only:
        exps = [e for e in exps if args.only in e['name']]
    if not exps:
        print('No experiments selected.'); sys.exit(0)

    parallel = args.parallel or len(args.gpus)
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join('runs', ts)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, 'summary_results.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['name', 'gpu', 'best_av_ap', 'best_v_ap', 'best_epoch', 'log'])
    # save experiment manifest
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump([{'name': e['name'], 'seed': e['seed'], 'params': e['params'], 'switches': e['switches']} for e in exps], f, indent=2)

    print(f'[{len(exps)} experiments] groups={groups} parallel={parallel} gpus={args.gpus}')
    print(f'results -> {csv_path}')

    mgr = Manager()
    gpu_queue = mgr.Queue()
    csv_lock = mgr.Lock()
    for g in args.gpus:
        gpu_queue.put(g)

    bundles = [(e, run_dir, gpu_queue, csv_path, csv_lock, args.dry) for e in exps]
    if args.dry or len(bundles) == 1:
        for b in bundles:
            run_one(b)
    else:
        with Pool(processes=parallel) as pool:
            pool.map(run_one, bundles)
    print(f'\nDone. Summary: {csv_path}')
