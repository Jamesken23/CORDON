"""Load TO / RE / TD / LE splits under ``data/``."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
CACHE_DIR = ROOT / "cache"

DATASET_DIR = {
    "RE": "RE",
    "TD": "TD",
    "LE": "LE",
    "TO": "TO",
}

DATASETS = ("TO", "RE", "TD", "LE")


def _digit_token(token: str) -> str:
    if "0x" in token or any(ch in token for ch in "123456789"):
        return "number"
    return token


def encode_split(json_path: Path, vocab: dict, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    pad = vocab["PAD"]
    with json_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids = []
            for token in obj["sol content"].split(" "):
                if token == "":
                    continue
                token = _digit_token(token)
                if token in vocab:
                    ids.append(vocab[token])
            if len(ids) < max_len:
                ids.extend([pad] * (max_len - len(ids)))
            else:
                ids = ids[:max_len]
            rows.append(ids)
            labels.append(int(obj["label"]))
    return np.asarray(rows, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def _folder(dataset: str) -> Path:
    if dataset not in DATASET_DIR:
        raise KeyError(f"Unknown dataset {dataset}. Choose from {DATASETS}.")
    return DATA_ROOT / DATASET_DIR[dataset]


def _vocab_path(folder: Path, dataset: str) -> Path:
    return folder / f"{DATASET_DIR[dataset]}_vocab_id.pkl"


def load_vocab_size(dataset: str) -> int:
    folder = _folder(dataset)
    with _vocab_path(folder, dataset).open("rb") as handle:
        vocab = pickle.load(handle)
    return int(max(vocab.values()) + 1)


def load_test(dataset: str, max_len: int = 2000) -> dict:
    """Load only the test split (no training file required)."""
    folder = _folder(dataset)
    with _vocab_path(folder, dataset).open("rb") as handle:
        vocab = pickle.load(handle)
    test_path = folder / "test_data.json"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")
    test_x, test_y = encode_split(test_path, vocab, max_len)
    return {
        "dataset": dataset,
        "x_test": test_x,
        "y_test": test_y,
        "vocab": int(max(vocab.values()) + 1),
    }


def stratified_split(train_x, train_y, label_rate: float, seed: int):
    rng = np.random.RandomState(seed)
    labeled_idx = []
    unlabeled_idx = []
    for cls in (0, 1):
        idx = np.where(train_y == cls)[0]
        rng.shuffle(idx)
        keep = int(len(idx) * label_rate)
        labeled_idx.append(idx[:keep])
        unlabeled_idx.append(idx[keep:])
    labeled_idx = np.concatenate(labeled_idx)
    unlabeled_idx = np.concatenate(unlabeled_idx)
    rng.shuffle(labeled_idx)
    rng.shuffle(unlabeled_idx)
    return (
        train_x[labeled_idx],
        train_y[labeled_idx],
        train_x[unlabeled_idx],
        train_y[unlabeled_idx],
    )


def load_split(dataset: str, max_len: int, label_rate: float, seed: int, le_subsample: int = 6):
    """Load train + test. Requires ``training_data.json`` under ``data/<DATASET>/``."""
    folder = _folder(dataset)
    train_path = folder / "training_data.json"
    test_path = folder / "test_data.json"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Place the training JSON next to test_data.json "
            f"(see README). For test-only evaluation use: python -m cordon.eval"
        )
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{dataset}_L{max_len}.npz"
    if cache_path.exists():
        blob = np.load(cache_path)
        train_x, train_y = blob["train_x"], blob["train_y"]
        test_x, test_y = blob["test_x"], blob["test_y"]
        vocab_size = int(blob["vocab"])
    else:
        with _vocab_path(folder, dataset).open("rb") as handle:
            vocab = pickle.load(handle)
        train_x, train_y = encode_split(train_path, vocab, max_len)
        test_x, test_y = encode_split(test_path, vocab, max_len)
        vocab_size = int(max(vocab.values()) + 1)
        np.savez_compressed(
            cache_path,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            vocab=np.int64(vocab_size),
        )

    if dataset == "LE" and le_subsample and le_subsample > 1:
        rng = np.random.RandomState(seed)
        take = max(2, len(train_x) // le_subsample)
        chosen = rng.choice(len(train_x), size=take, replace=False)
        train_x = train_x[chosen]
        train_y = train_y[chosen]

    x_l, y_l, x_u, y_u = stratified_split(train_x, train_y, label_rate, seed)
    return {
        "dataset": dataset,
        "x_train": train_x,
        "y_train": train_y,
        "x_l": x_l,
        "y_l": y_l,
        "x_u": x_u,
        "y_u": y_u,
        "x_test": test_x,
        "y_test": test_y,
        "vocab": vocab_size,
    }
