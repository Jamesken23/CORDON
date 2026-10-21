"""Train CORDON on TO / RE / TD / LE with paper defaults (needs training_data.json)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from cordon.train import run

DATASETS = ("TO", "RE", "TD", "LE")
ROOT = Path(__file__).resolve().parent


def finished(metric_csv: Path) -> set[str]:
    if not metric_csv.exists():
        return set()
    with metric_csv.open("r", encoding="utf-8") as handle:
        return {row["dataset"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CORDON on all four paper datasets.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    args_cli = parser.parse_args()

    import torch

    device = args_cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results"
    ckpt_dir = ROOT / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = finished(out_dir / "metrics.csv")

    for dataset in DATASETS:
        if dataset in done:
            print("skip", dataset, flush=True)
            continue
        train_path = ROOT / "data" / dataset / "training_data.json"
        if not train_path.exists():
            print(
                f"SKIP {dataset}: missing {train_path}. "
                "Place training_data.json to train; test-only eval needs a checkpoint.",
                flush=True,
            )
            continue
        args = SimpleNamespace(
            dataset=dataset,
            label_rate=0.2,
            seed=0,
            max_len=2000,
            embedding_dim=256,
            dropout=0.5,
            batch_size=8,
            lr=1e-3,
            epochs=args_cli.epochs,
            warmup=20,
            remember_rate=0.6,
            le_subsample=6,
            pseudo_confidence=0.8,
            adv_coef=0.1,
            unaudited_ramp_epochs=10,
            plsae_ramp_length=20,
            device=device,
            out_dir=str(out_dir),
            ckpt_dir=str(ckpt_dir),
        )
        print("START CORDON", dataset, flush=True)
        record = run(args)
        print(
            "DONE",
            dataset,
            "bestF1",
            round(record["test_f1"], 4),
            "epoch",
            record["best_epoch"],
            flush=True,
        )


if __name__ == "__main__":
    main()
