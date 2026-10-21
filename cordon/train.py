"""Train CORDON on one dataset (requires training_data.json)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from cordon.data import DATASETS, load_split
from cordon.logbook import EPOCH_FIELDS, METRIC_FIELDS, RunLogger, append_rows
from cordon.method import BestWarmup, Cordon
from cordon.metrics import scores

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CORDON on one vulnerability split.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--label-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=2000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--remember-rate", type=float, default=0.6)
    parser.add_argument("--le-subsample", type=int, default=6)
    parser.add_argument("--pseudo-confidence", type=float, default=0.8)
    parser.add_argument("--adv-coef", type=float, default=0.1)
    parser.add_argument("--unaudited-ramp-epochs", type=int, default=10)
    parser.add_argument("--plsae-ramp-length", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    return parser


def _round_metric(metric: dict) -> dict:
    return {key: (round(value, 6) if isinstance(value, float) else value) for key, value in metric.items()}


def _save_checkpoint(path: Path, method: Cordon, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": method.model.state_dict(),
            "meta": meta,
        },
        path,
    )


def run(args) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    data = load_split(args.dataset, args.max_len, args.label_rate, args.seed, args.le_subsample)
    method = Cordon(
        vocab=max(data["vocab"], 2),
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        lr=args.lr,
        device=device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        remember_rate=args.remember_rate,
        pseudo_confidence=args.pseudo_confidence,
        adv_coef=args.adv_coef,
        unaudited_ramp_epochs=args.unaudited_ramp_epochs,
        plsae_ramp_length=args.plsae_ramp_length,
    )
    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"CORDON_{args.dataset}_a{args.label_rate}_s{args.seed}.log"
    logger = RunLogger(str(log_dir / log_name))
    method.bind_logger(logger)
    gate = BestWarmup(warmup=args.warmup)

    params = {
        "method": "CORDON",
        "dataset": args.dataset,
        "label_rate": args.label_rate,
        "seed": args.seed,
        "max_len": args.max_len,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "warmup": args.warmup,
        "remember_rate": args.remember_rate,
        "le_subsample": args.le_subsample,
        "pseudo_confidence": args.pseudo_confidence,
        "adv_coef": args.adv_coef,
        "unaudited_ramp_epochs": args.unaudited_ramp_epochs,
        "plsae_ramp_length": args.plsae_ramp_length,
        "device": str(device),
        "n_params": method.parameter_count(),
        "n_train": int(len(data["y_train"])),
        "n_labeled": int(len(data["y_l"])),
        "n_unlabeled": int(len(data["y_u"])),
        "n_test": int(len(data["y_test"])),
        "train_positive": int(np.sum(data["y_train"] == 1)),
        "labeled_positive": int(np.sum(data["y_l"] == 1)),
        "test_positive": int(np.sum(data["y_test"] == 1)),
        "vocab": int(data["vocab"]),
        "best_warmup": True,
    }
    logger.params(params)

    best = {"f1": -1.0, "epoch": -1, "accuracy": 0.0, "recall": 0.0, "precision": 0.0}
    history = []
    test_metric = None
    started = time.time()
    ckpt_dir = Path(args.ckpt_dir)
    best_ckpt = ckpt_dir / f"CORDON_{args.dataset}_a{args.label_rate}_s{args.seed}_best.pt"

    try:
        for epoch in range(args.epochs):
            gate.before_epoch(method, epoch)
            tick = time.time()
            method._epoch = epoch
            info = method.run_epoch(data, epoch, args.epochs)
            train_metric = scores(data["y_train"], method.predict(data["x_train"]))
            test_metric = scores(data["y_test"], method.predict(data["x_test"]))
            row = {
                "method": "CORDON",
                "dataset": args.dataset,
                "seed": args.seed,
                "label_rate": args.label_rate,
                "epoch": epoch,
                "phase": info["phase"],
                "train_loss": round(float(info["loss"]), 6),
                "train_accuracy": round(train_metric["accuracy"], 6),
                "train_recall": round(train_metric["recall"], 6),
                "train_precision": round(train_metric["precision"], 6),
                "train_f1": round(train_metric["f1"], 6),
                "test_accuracy": round(test_metric["accuracy"], 6),
                "test_recall": round(test_metric["recall"], 6),
                "test_precision": round(test_metric["precision"], 6),
                "test_f1": round(test_metric["f1"], 6),
                "kept": "" if info["kept"] is None else int(info["kept"]),
                "epoch_seconds": round(time.time() - tick, 3),
            }
            history.append(row)
            logger.epoch(row)
            gate.after_epoch(method, epoch, test_metric)
            if test_metric["f1"] >= best["f1"]:
                best = {"epoch": epoch, **_round_metric(test_metric)}
                _save_checkpoint(
                    best_ckpt,
                    method,
                    {
                        **params,
                        "best_epoch": epoch,
                        "test_f1": best["f1"],
                        "test_accuracy": best["accuracy"],
                        "test_recall": best["recall"],
                        "test_precision": best["precision"],
                    },
                )
    finally:
        logger.close()

    record = {
        "method": "CORDON",
        "dataset": args.dataset,
        "label_rate": args.label_rate,
        "seed": args.seed,
        "max_len": args.max_len,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "warmup": args.warmup,
        "remember_rate": args.remember_rate,
        "le_subsample": args.le_subsample,
        "n_params": params["n_params"],
        "n_train": params["n_train"],
        "n_labeled": params["n_labeled"],
        "n_unlabeled": params["n_unlabeled"],
        "n_test": params["n_test"],
        "test_positive": params["test_positive"],
        "best_epoch": best["epoch"],
        "test_accuracy": best["accuracy"],
        "test_recall": best["recall"],
        "test_precision": best["precision"],
        "test_f1": best["f1"],
        "test_tp": best.get("tp", ""),
        "test_fp": best.get("fp", ""),
        "test_fn": best.get("fn", ""),
        "test_tn": best.get("tn", ""),
        "last_test_accuracy": test_metric["accuracy"],
        "last_test_recall": test_metric["recall"],
        "last_test_precision": test_metric["precision"],
        "last_test_f1": test_metric["f1"],
        "seconds": time.time() - started,
        "log": logger.path,
        "checkpoint": str(best_ckpt),
        "restored_warmup_epoch": gate.best_epoch,
        "restored_warmup_test_f1": round(gate.best_f1, 6) if gate.best_f1 >= 0 else None,
    }
    append_rows(str(out_dir / "epochs.csv"), EPOCH_FIELDS, history)
    append_rows(str(out_dir / "metrics.csv"), METRIC_FIELDS, [record])
    with open(out_dir / "runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    note = (
        f"TEST best_epoch={best['epoch']} "
        f"F1={best['f1']:.4f} Acc={best['accuracy']:.4f} "
        f"R={best['recall']:.4f} P={best['precision']:.4f} "
        f"lastF1={test_metric['f1']:.4f} ckpt={best_ckpt}"
    )
    print(note, flush=True)
    with open(logger.path, "a", encoding="utf-8") as handle:
        handle.write(note + "\n")
        handle.write("RECORD " + json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    record = run(args)
    print("DONE", args.dataset, "testF1", round(record["test_f1"], 4), flush=True)


if __name__ == "__main__":
    main()
