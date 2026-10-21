"""Text log and CSV helpers for one training run."""

from __future__ import annotations

import csv
import json
import os


EPOCH_FIELDS = [
    "method", "dataset", "seed", "label_rate", "epoch", "phase",
    "train_loss", "train_accuracy", "train_recall", "train_precision", "train_f1",
    "test_accuracy", "test_recall", "test_precision", "test_f1", "kept", "epoch_seconds",
]

METRIC_FIELDS = [
    "method", "dataset", "seed", "label_rate", "max_len", "embedding_dim", "dropout",
    "batch_size", "lr", "epochs", "warmup", "remember_rate", "le_subsample",
    "n_params", "n_train", "n_labeled", "n_unlabeled", "n_test", "test_positive",
    "best_epoch",
    "test_accuracy", "test_recall", "test_precision", "test_f1",
    "test_tp", "test_fp", "test_fn", "test_tn",
    "last_test_accuracy", "last_test_recall", "last_test_precision", "last_test_f1",
    "seconds",
]


class RunLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.handle = open(path, "w", encoding="utf-8")
        self.steps = 0

    def params(self, payload: dict) -> None:
        self._line("PARAMS " + json.dumps(payload, ensure_ascii=False), echo=True)

    def step(self, epoch: int, loss: float) -> None:
        self.steps += 1
        self.handle.write(f"STEP epoch={epoch} step={self.steps} loss={loss:.6f}\n")
        if self.steps % 50 == 0:
            self.handle.flush()

    def epoch(self, payload: dict) -> None:
        self.handle.flush()
        self._line("EPOCH " + json.dumps(payload, ensure_ascii=False), echo=True)

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            self.handle.close()

    def _line(self, text: str, echo: bool) -> None:
        self.handle.write(text + "\n")
        self.handle.flush()
        if echo:
            print(text, flush=True)


def append_rows(path: str, fields: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
