# MGCIL

**Multi-Granular Contrastive Learning for Lightweight Weakly-Supervised Audio-Visual Violence Detection**

Chun-Guo Li, Shuai Xie, Xiaoxiao Liu, Yidan Wang

*Hebei Key Laboratory of Machine Learning and Computational Intelligence, College of Mathematics and Information Science, Hebei University, Baoding 071002, China*

Official PyTorch implementation. Manuscript under review.

---

## Overview

MGCIL upgrades the MACIL-SD backbone by organizing contrastive supervision at **three granularities**, without adding a single trainable parameter (0.678M, identical to the backbone).

| Level | Module | What it changes |
|---|---|---|
| Segment | **CGM** — Confidence-Gated semi-bag Mining | Replaces the hard 0.5 threshold with a soft confidence gate, so a video without a confident side is diluted across both pools instead of being committed to one |
| Segment | **ANC** — Adversarial Negative Construction | Builds worst-case negatives by one-step FGSM perturbation instead of random sampling |
| Segment | **UWC** — Uncertainty-Weighted Cross-modal Consistency | Down-weights the audio–visual agreement on low-confidence segments |
| Semantic | **SAO** — Semantic Anchor Orthogonalization | Anchors normal/abnormal representations to frozen CLIP text priors made mutually orthogonal by Gram–Schmidt (0 learnable parameters) |
| Video | **VLC** — Video-Level Contrastive | Pulls whole abnormal videos together and away from normal videos |
| Inference | **GTS** — Gaussian Temporal Smoothing | Suppresses isolated single-frame score spikes (σ = 2) |

## Results on XD-Violence

| Method | Modality | AP (%) | Params |
|---|---|---|---|
| MACIL-SD (reproduced in our environment) | Audio & Visual | 82.90 | 0.678M |
| MGS (previous work of this group, hyperbolic) | Audio & Visual | 87.37 | 0.855M |
| **MGCIL (ours)** | Audio & Visual | **88.58** | **0.678M** |

## Repository structure

```
MGCIL_release/
├── main.py                  # training entry
├── option.py                # all hyper-parameters and module switches (--use_*)
├── infer.py                 # evaluation entry
├── avce_network.py          # backbone + MGCIL modules
├── avce_dataset.py          # XD-Violence feature dataset
├── train.py / test.py       # training / evaluation loops
├── InfoNCE.py               # contrastive losses
├── CMA_MIL.py               # cross-modal attention MIL head
├── Transformer.py           # temporal encoder
├── mixed_manifold.py        # mixed-manifold InfoNCE (regime A)
├── extract_clip.py          # builds the frozen CLIP text anchors (run once)
├── run_experiments.py       # ablation / sensitivity / multi-seed driver
├── report_results.py        # aggregates runs/*/summary_results.csv
├── plot_sensitivity.py      # Fig. 3-4 of the paper
├── gen_qualitative.py       # Fig. 5 of the paper
├── gen_tsne.py              # Fig. 6 of the paper
├── fig1_motivation.py       # Fig. 1 of the paper
├── fig2_pipeline.py         # Fig. 2 of the paper
├── anchor_abn_clip.pt       # frozen CLIP text anchors (2 KB)
├── anchor_nor_clip.pt       # frozen CLIP text anchors (2 KB)
├── list/                    # feature path lists and test-set ground truth
├── ckpt/full_s1888.pkl      # main model of the paper (seed 1888)
└── MGCIL_model_reference.md # full technical reference: equations, gradients, loss scales
```

## Environment

```
python >= 3.7
torch >= 1.6        (CUDA recommended for training)
numpy, scikit-learn, matplotlib
```

The reported results were obtained on a single GPU. Different GPU types and torch/CUDA versions can change the numbers slightly; the paper fixes the seed (`--seed 1888`) to keep runs comparable.

## Data preparation

Download **RGB.zip**, **RGBTest.zip** and **vggish-features.zip** from the
[XD-Violence benchmark](https://roc-ng.github.io/XD-Violence/), unzip them into `data/`,
and keep the file lists in `list/` (`rgb.list`, `audio.list`, `rgb_test.list`,
`audio_test.list`, `gt.npy`). Regenerate the lists with `list/make_list.py` and
`list/make_gt.py` if your directory layout differs.

The frozen CLIP text anchors in `anchor_*.pt` are already provided. To rebuild them
from scratch (requires `clip` and a CLIP checkpoint):

```bash
python extract_clip.py
```

## Training

```bash
# full MGCIL (this is the default: every --use_* switch is 1)
python main.py --model_name mgcil --gpus 0
```

All six modules are controlled by integer switches, so any ablation in Table 4 can be
reproduced from the command line:

```bash
# example: drop SAO only
python main.py --use_anchor 0

# reproduce the pure MACIL-SD baseline
python main.py --use_cgm 0 --use_uwc 0 --use_anchor 0 --use_adv 0 \
               --use_vlc 0 --use_gts 0 --grad_clip 0 --model_name macil_sd
```

Key hyper-parameters (see `option.py` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--tau` | 0.1 | InfoNCE temperature τ |
| `--lamda_anchor` | 0.1 | weight of the semantic-anchor loss λ_sa |
| `--lamda_video` | 0.1 | weight of the video-level loss λ_vlc |
| `--adv_eps` | 0.005 | FGSM step for adversarial negatives |
| `--conf_temp` | 0.2 | CGM gating temperature τ_c |
| `--gts_sigma` | 2.0 | Gaussian smoothing σ at inference |
| `--seed` | 1888 | random seed |

## Evaluation

```bash
python infer.py --model_dir ckpt/full_s1888.pkl
```

`ckpt/full_s1888.pkl` is the run reported in the paper (88.58% AP). The ablation and
sensitivity checkpoints are not shipped; regenerate them with `run_experiments.py`.

## Citation

```bibtex
@misc{li2026mgcil,
  title  = {Multi-Granular Contrastive Learning for Lightweight
            Weakly-Supervised Audio-Visual Violence Detection},
  author = {Li, Chun-Guo and Xie, Shuai and Liu, Xiaoxiao and Wang, Yidan},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant No. 62506114)
and the Natural Science Foundation of Hebei Province (Grant Nos. A2024201031 and H2024201062).

The codebase builds on [XDVioDet](https://github.com/Roc-Ng/XDVioDet),
[RTFM](https://github.com/tianyu0207/RTFM) and MACIL-SD (Yu et al., ACM MM 2022).
We thank the authors for releasing their code.

## License

MIT. See [LICENSE](LICENSE).
