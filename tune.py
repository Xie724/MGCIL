import os
import subprocess
import itertools
import datetime
import csv
import re
import sys
from multiprocessing import Pool, Manager

# ==============================================================================
# --- 1. 定义你要测试的超参数网格 (根据你的 option.py 定制) ---
# ==============================================================================

param_grid = {
    'lr':           [0.0006],
    'dropout':      [0.4],
    'tau':          [0.14],                        # 锁定
    'lamda_a2b':    [0.6, 0.7, 0.8],              # 围住0.7
    'lamda_a2n':    [0.07, 0.11, 0.15],           # 首次为此版本扫
    'lamda_anchor': [0.025],
    'lamda_cof':    [0.06],
    'm':            [0.86, 0.87, 0.88],           # 首次为此版本扫
    'adv_eps':      [0.002, 0.003],               # 首次为此版本扫
    'lamda_video':  [0.01],                        # 锁定
    'batch-size':   [128],
    'max-epoch':    [50],
}
# 3×3×3×2 = 54 组 — 扫辅助参数

# ==============================================================================
# --- 2. 定义其他固定的训练参数 ---
# ==============================================================================
fixed_args = [
    '--modality', 'MIX2',
    '--batch-size', '128',
    '--workers', '0',
    '--max-epoch', '50',
    '--rgb-list', 'list/rgb.list',
    '--audio-list', 'list/audio.list',
    '--test-rgb-list', 'list/rgb_test.list',
    '--test-audio-list', 'list/audio_test.list',
    '--gt', 'list/gt.npy',
    '--hid_dim', '128',
    '--ffn_dim', '128',
    '--num_stages', '3',
    '--nhead', '4'
]

# ==============================================================================
# --- 3. 并行设置 (根据你的硬件配置修改) ---
# ==============================================================================
# 设置你想同时进行的进程个数
num_parallel_processes = 4  # 建议根据显卡显存和数量调整

# 列出你可用的 GPU ID
available_gpus = ['1','1','1','1']  # 如果你只有单卡，写成 ['0', '0'] 跑两个进程，或者 ['0'] 跑一个

# ==============================================================================
# --- 4. 实验准备 ---
# ==============================================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base_run_dir = os.path.join('tuning_runs', timestamp)
os.makedirs(base_run_dir, exist_ok=True)

results_csv_path = os.path.join(base_run_dir, 'summary_results.csv')
param_keys = list(param_grid.keys())
# 表头修改为适应 main.py 的输出: best_av_auc, best_v_auc
csv_header = ['trial_id', 'gpu_id'] + param_keys + ['best_av_ap', 'best_v_ap', 'best_epoch']

with open(results_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)


# ==============================================================================
# --- 5. 定义单个实验的执行函数 ---
# ==============================================================================
def run_trial(args_bundle):
    i, params, base_run_dir, gpu_queue, csv_path, csv_lock = args_bundle
    gpu_id = gpu_queue.get()

    trial_id = f"trial_{i + 1:03d}"
    print(f"\n{'=' * 20}\nStarting {trial_id} on GPU {gpu_id}: {params}\n{'=' * 20}")

    trial_dir = os.path.join(base_run_dir, trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    # 动态构建命令
    python_executable = sys.executable
    command = [python_executable, 'main.py'] + fixed_args

    # 【关键修改】：适配你 main.py 中的参数解析方式
    command.extend(['--gpus', str(gpu_id)])

    # 动态添加调参网格中的参数
    for key, value in params.items():
        command.extend([f'--{key}', str(value)])

    # 为了防止多个 trial 覆盖同一个 ckpt，我们给它们加上 trial_id 的后缀
    model_name_for_this_trial = f'macil_sd_{trial_id}'
    command.extend(['--model-name', model_name_for_this_trial])

    log_path = os.path.join(trial_dir, 'training.log')
    param_values = list(params.values())
    result_row = [trial_id, gpu_id] + param_values + ['FAILED', 'FAILED', 'FAILED']

    try:
        # 运行子进程
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8')

        # 实时写入日志文件
        with open(log_path, 'w') as log_file:
            for line in process.stdout:
                # 可以选择把输出打印到控制台，如果并行数多，建议注释掉下一行以免刷屏
                # print(f"[{trial_id} on GPU {gpu_id}] {line.strip()}")
                log_file.write(line)

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        # 实验成功后，解析日志
        with open(log_path, 'r') as log_file:
            full_log = log_file.read()

        best_epoch, best_av_auc, best_v_auc = 'N/A', 'N/A', 'N/A'

        # 【关键修改】：适配你 main.py 最终输出的日志格式
        # 你的日志: Best Performance in Epoch 25: av_auc:0.8750 | v_auc:0.8500
        match = re.search(r"Best Performance in Epoch (\d+):\s*av_ap:([\d.]+)\s*\|\s*v_ap:([\d.]+)", full_log)

        if match:
            best_epoch = match.group(1)
            best_av_auc = f"{float(match.group(2)):.4f}"
            best_v_auc = f"{float(match.group(3)):.4f}"
            result_row = [trial_id, gpu_id] + param_values + [best_av_auc, best_v_auc, best_epoch]
        else:
            print(f"!!!!!! {trial_id} on GPU {gpu_id} finished, but couldn't parse the result line! !!!!!!")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"!!!!!! {trial_id} on GPU {gpu_id} FAILED !!!!!!\nError: {e}")
    finally:
        gpu_queue.put(gpu_id)
        # 立即写入 CSV，防止中途崩溃丢失所有结果
        with csv_lock:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_row)
        print(f"--- Finished {trial_id} on GPU {gpu_id}. GPU {gpu_id} is now free. ---")
        return result_row


# ==============================================================================
# --- 6. 生成并运行所有实验 ---
# ==============================================================================
if __name__ == '__main__':
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"将要进行 {len(param_combinations)} 次实验...")
    print(f"使用 {num_parallel_processes} 个并行进程.")
    print(f"可用 GPUs: {available_gpus}")
    print(f"所有结果将保存在: {base_run_dir}")

    manager = Manager()
    gpu_queue = manager.Queue()
    csv_lock = manager.Lock()
    for gpu in available_gpus:
        gpu_queue.put(gpu)

    tasks = [(i, params, base_run_dir, gpu_queue, results_csv_path, csv_lock)
             for i, params in enumerate(param_combinations)]

    with Pool(processes=num_parallel_processes) as pool:
        pool.map(run_trial, tasks)

    print(f"\n所有实验完成！总结报告见: {results_csv_path}")