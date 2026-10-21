# CORDON

Official reproduction code for **CORDON**.

CORDON detects smart-contract vulnerabilities under weak auditing: a confidence gate (τ=0.8), an anchored class cordon, PL-style paired labeled/unlabeled steps, and BestWarmup (restore the best warm-up checkpoint before the cordon phase).


## Datasets

| Name | Vulnerability | Folder |
|------|---------------|--------|
| TO   | Tx-origin     | `data/TO/` |
| RE   | Reentrancy    | `data/RE/` |
| TD   | Timestamp dependency | `data/TD/` |
| LE   | Locked ether  | `data/LE/` |

Each folder contains:

- `test_data.json` — test split (JSONL; each line has `sol content` and `label`)
- `<NAME>_vocab_id.pkl` — token vocabulary

Training requires you to place `training_data.json` next to the test file, e.g.:

```
data/TO/training_data.json
data/RE/training_data.json
data/TD/training_data.json
data/LE/training_data.json
```

The JSONL schema matches the test files (`sol content`, `label`).

## Requirements

- Python 3.9+
- CUDA GPU recommended (CPU works but is slow)

```bash
pip install -r requirements.txt
```

Run all commands from the repository root (`CORDON-release/`).

## Quick start: evaluate a checkpoint

If you already have a checkpoint produced by this code:

```bash
python -m cordon.eval --checkpoint checkpoints/CORDON_TO_a0.2_s0_best.pt
```

Optional flags:

```bash
python -m cordon.eval \
  --checkpoint checkpoints/CORDON_RE_a0.2_s0_best.pt \
  --dataset RE \
  --device cuda
```

Output is Acc / Recall / Precision / F1 on the shipped test split.


**One dataset** (needs `data/<DATASET>/training_data.json`):

```bash
python -m cordon.train --dataset TO --device cuda
```

Other datasets: `--dataset RE`, `--dataset TD`, `--dataset LE`.

**All four datasets:**

```bash
python run_all.py --device cuda
```


## Project layout

```
CORDON-release/
├── README.md
├── requirements.txt
├── run_all.py
├── cordon/
│   ├── __init__.py
│   ├── __main__.py
│   ├── data.py          # load test / train+test
│   ├── model.py         # ConvMHSA
│   ├── select.py        # anchored cordon
│   ├── method.py        # CORDON + BestWarmup
│   ├── metrics.py
│   ├── logbook.py
│   ├── train.py
│   └── eval.py
├── data/
│   ├── TO/  (test + vocab)
│   ├── RE/
│   ├── TD/
│   └── LE/
├── checkpoints/         # created by training
└── results/             # created by training
```

## Citation

If you use this code, please cite the Defence Technology paper describing CORDON.
