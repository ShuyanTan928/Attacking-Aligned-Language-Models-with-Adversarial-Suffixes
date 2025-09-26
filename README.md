# Attacking-Aligned-Language-Models-with-Adversarial-Suffixes

This repository contains a lightweight implementation of greedy coordinate gradient (GCG) attacks that optimize adversarial suffixes against aligned language models. It reproduces the two tasks described in the prompt:

1. **Task 1 – Single-model optimization:** learn a suffix that causes the Qwen/Qwen3-0.6B model to emit a targeted hazardous response.
2. **Task 2 – Transferability via model ensembling:** optimize the suffix jointly against Qwen/Qwen3-0.6B (primary) and meta-llama/Llama-3.2-1B-Instruct (auxiliary) to encourage transfer.

Both tasks log loss values, export SVG plots every 20 iterations by default, and save example prompts, suffixes, and model generations.

## Repository layout

- `attacks/gcg.py` – Core attack implementation including model wrappers, the GCG optimizer, and CSV logging utility.
- `scripts/run_task1.py` – Driver for Task 1 single-model optimization against Qwen.
- `scripts/run_task2.py` – Driver for Task 2 ensemble optimization with Qwen (primary) and Llama (auxiliary).
- `utils/simple_plot.py` – Pure-Python SVG plotting helper used to render loss curves without external plotting libraries.
- `results/` – Output directory created by the task scripts (not tracked in git).

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/) with CUDA support recommended for faster optimization.
- [transformers](https://huggingface.co/docs/transformers/index) (tested with >=4.40).
- [huggingface-hub](https://huggingface.co/docs/huggingface_hub/index) for model downloads (installed alongside `transformers`).

Install the dependencies in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers huggingface-hub
```

> **Note:** The exact PyTorch install command depends on your platform and CUDA version. Refer to the [official instructions](https://pytorch.org/get-started/locally/).

Before running the scripts, make sure you can authenticate with Hugging Face Hub if your environment requires it (`huggingface-cli login`).

## Usage

All scripts default to CPU if CUDA is unavailable. Use `--device cuda` to force GPU execution when possible. The `--log-every` argument controls logging frequency; the default `20` aligns with the deliverable specification.

### Task 1 – Single-model optimization

Run the Task 1 driver to optimize a suffix solely against Qwen/Qwen3-0.6B:

```bash
python scripts/run_task1.py \
    --iterations 120 \
    --suffix-length 24 \
    --top-k 40 \
    --log-every 20 \
    --device cuda  # omit or change to "cpu" as needed
```

Outputs are written to `results/task1/` by default (override with `--output-dir`). The script generates:

- `optimization_log.csv` – loss snapshots `(iteration, loss)` recorded at initialization and every `log_every` iterations.
- `task1_loss_curve.svg` – smooth SVG line plot of the loss history.
- `task1_suffix_and_output.txt` – optimized suffix followed by the model's greedy generation.
- `task1_prompt.txt` – final adversarial prompt presented to the model.

### Task 2 – Transferability via ensembling

Launch the Task 2 driver to optimize a suffix that transfers between Qwen and Llama:

```bash
python scripts/run_task2.py \
    --iterations 100 \
    --suffix-length 24 \
    --top-k 40 \
    --log-every 20 \
    --device cuda
```

The default outputs live under `results/task2/` and include:

- `optimization_log.csv` – ensemble-averaged loss snapshots.
- `task2_loss_curve.svg` – loss curve plot mirroring Task 1.
- `task2_suffix_and_outputs.txt` – optimized suffix plus greedy generations from both Qwen and Llama.
- `task2_prompt.txt` – final ensemble prompt.

## Experiment customization

- **Prompt/target text:** Edit the `base_prompt`, `post_prompt`, and `target_response` strings near the top of each task script.
- **Suffix length and search breadth:** Adjust `--suffix-length` and `--top-k` to trade off runtime and search coverage.
- **Random seed:** Modify the `random_seed` argument passed to `GCGOptimizer` for different initializations.
- **Additional auxiliary models:** For Task 2, append more `AuxiliaryModelWrapper` instances to the optimizer to study broader transferability.

## Re-running & reproducibility

Each script is deterministic given a fixed random seed, model checkpoint, and hardware setup. If you resume an experiment, delete or rename the existing results directory to avoid mixing outputs. The CSV logs and SVG plots are intended for inclusion in reports or as deliverables.

## Troubleshooting

- **Missing PyTorch:** Install a compatible PyTorch build as described above. Without it, the scripts cannot run (the import will fail).
- **CUDA out-of-memory:** Lower the batch size by reducing `--suffix-length`, or fall back to CPU execution (`--device cpu`).
- **Slow downloads:** Pre-cache the Hugging Face models using `transformers-cli download` or mirror them locally.

## License

The underlying models are distributed under their respective licenses via Hugging Face Hub. Refer to the upstream model cards for usage constraints.
