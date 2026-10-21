"""Evaluate a saved CORDON checkpoint on the shipped test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cordon.data import DATASETS, load_test
from cordon.method import Cordon
from cordon.metrics import scores

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CORDON checkpoint on test data.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt saved by train.py")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Override dataset name (default: read from checkpoint meta).",
    )
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    blob = torch.load(ckpt_path, map_location="cpu")
    meta = blob.get("meta", {})
    dataset = args.dataset or meta.get("dataset")
    if dataset not in DATASETS:
        raise ValueError(f"Need a valid --dataset (got {dataset!r}). Choose from {DATASETS}.")
    max_len = args.max_len or int(meta.get("max_len", 2000))
    embedding_dim = int(meta.get("embedding_dim", 256))
    dropout = float(meta.get("dropout", 0.5))
    batch_size = args.batch_size or int(meta.get("batch_size", 8))
    vocab = int(meta.get("vocab", 0))

    data = load_test(dataset, max_len=max_len)
    if vocab <= 0:
        vocab = data["vocab"]
    device = torch.device(args.device)
    method = Cordon(
        vocab=max(vocab, 2),
        embedding_dim=embedding_dim,
        dropout=dropout,
        lr=1e-3,
        device=device,
        batch_size=batch_size,
    )
    method.model.load_state_dict(blob["model"])
    metric = scores(data["y_test"], method.predict(data["x_test"]))
    out = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset,
        "n_test": int(len(data["y_test"])),
        "accuracy": round(metric["accuracy"], 6),
        "recall": round(metric["recall"], 6),
        "precision": round(metric["precision"], 6),
        "f1": round(metric["f1"], 6),
        "tp": metric["tp"],
        "fp": metric["fp"],
        "fn": metric["fn"],
        "tn": metric["tn"],
        "meta_best_epoch": meta.get("best_epoch"),
        "meta_test_f1": meta.get("test_f1"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
