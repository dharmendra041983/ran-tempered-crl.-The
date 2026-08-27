# Tempered CRL for Multi-Cell NG-RAN

This repository contains the code used for the multi-cell NG-RAN experiments on SLA-aware admission and resource allocation with per-slice constrained reinforcement learning.

The simulator uses seven cells and three traffic slices. The main comparison includes raw dual ascent, the tempered update, and full budget normalization. The repository also includes the adapted CMARS-inspired baseline, unseen-condition evaluation, and the CPU inference benchmark.

## Setup

Python 3.11 was used for the reported experiments.

```bash
pip install -r requirements.txt
```

## Primary experiment

The three variants are `raw`, `tempered`, and `full`, corresponding to alpha values 0, 0.5, and 1.

```bash
python run_variants.py --stage all --seeds 0 1 2 3 4 5 6 7 8 9 --steps 90000 --eval-seeds 5 --episodes 10
python validate_variants.py
python analyze_variants.py --seeds 0 1 2 3 4 5 6 7 8 9 --bootstrap 10000
```

Outputs are written under `results/primary/`. The repository includes the evaluation outputs used for the reported 10-seed comparison; trained model checkpoints are not included.

## CMARS-inspired baseline

The baseline follows the sequential slice-agent and constrained cooperative PPO structure of the CMARS work, adapted to the action and system model used here. It is not intended as an exact reproduction of the original implementation.

```bash
python run_cmars_adapted.py --stage all --seeds 0 1 2 3 4 --macro-steps 90000 --eval-seeds 5 --episodes 10
python validate_cmars_adapted.py
python analyze_cmars_adapted.py --seeds 0 1 2 3 4 --primary-root results/primary
```

Outputs are written under `results/cmars_adapted/`. The included results contain the five matched baseline seeds used in the comparison.

## Generalization evaluation

The generalization scripts evaluate trained policies without retraining under six operating shifts: load -15%, load +15%, stress factors 1.30 and 1.90, and per-cell capacity changes of -10% and +10%.

```bash
python run_generalization.py --primary-root results/primary --bootstrap 10000
```

Outputs are written under `results/generalization/`. The repository includes the episode-level unseen-condition evaluations and the corresponding summaries and statistical outputs.

## Inference benchmark

The benchmark measures deterministic Stable-Baselines3 policy inference and the projected dual-update kernel. Replace the model path with a trained tempered policy on your machine.

```powershell
python benchmark_inference.py --model "results\primary\tempered\seed0\crl_per_slice.zip" --runs 20000 --warmup 2000
```

The script writes `inference_timing.json` with latency and runtime information. The measured CPU-only benchmark used in the study is included under `results/inference/`.

## Files

- `multicell_env.py`: seven-cell environment
- `policies.py`: per-slice constrained SAC callback and dual updates
- `train_variants.py`, `evaluate_variants.py`, `analyze_variants.py`: primary experiment
- `cmars_adapted_env.py` and `*_cmars_adapted.py`: adapted multi-agent baseline
- `*_generalization.py`: unseen-condition evaluation
- `benchmark_inference.py`: CPU inference timing
- `configs/`: experiment settings
- `results/primary/`: 10-seed evaluation data for raw, tempered, and fully normalized dual updates
- `results/cmars_adapted/`: five-seed CMARS-inspired baseline evaluation data
- `results/generalization/`: unseen-condition episode data, summaries, and tests
- `results/inference/`: CPU-only inference timing output

Training-seed identity is treated as the independent replication unit. Evaluation episodes are paired across controller variants where comparisons require pairing.
