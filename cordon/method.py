"""CORDON trainer. Paper protocol: Conf80 + PL paired steps + BestWarmup."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from cordon.model import ConvMHSA
from cordon.select import anchored_cordon


def _loader(x, y, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(x, dtype=torch.long),
        torch.as_tensor(y, dtype=torch.long),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _rampup(epoch: int, length: int = 20) -> float:
    if epoch >= length:
        return 1.0
    phase = 1.0 - epoch / length
    return float(np.exp(-5.0 * phase * phase))


def _unit(v: torch.Tensor) -> torch.Tensor:
    denom = v.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    return v / denom


def unaudited_ramp(epoch: int, warmup: int, ramp_epochs: int = 10) -> float:
    if epoch < warmup:
        return 0.0
    step = epoch - warmup
    if step >= ramp_epochs:
        return 1.0
    return step / float(ramp_epochs)


class Cordon:
    """CORDON = Conf80 + anchored cordon + PL steps + BestWarmup (paper method)."""

    name = "CORDON"
    lab_batch = 8
    unlab_batch = 32

    def __init__(
        self,
        vocab: int,
        embedding_dim: int,
        dropout: float,
        lr: float,
        device: torch.device,
        batch_size: int,
        warmup: int = 20,
        remember_rate: float = 0.6,
        w1: float = 0.5,
        w2: float = 0.5,
        temperature: float = 0.1,
        eps: float = 0.5,
        pseudo_confidence: float = 0.8,
        adv_coef: float = 0.1,
        unaudited_ramp_epochs: int = 10,
        plsae_ramp_length: int = 20,
        **_extra,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = ConvMHSA(vocab, embedding_dim, dropout=dropout).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.ce = torch.nn.CrossEntropyLoss()
        self._logger = None
        self._epoch = 0

        self.warmup = warmup
        self.remember_rate = remember_rate
        self.w1 = w1
        self.w2 = w2
        self.temperature = temperature
        self.eps = eps
        self.pseudo_confidence = pseudo_confidence
        self.adv_coef = adv_coef
        self.unaudited_ramp_epochs = unaudited_ramp_epochs
        self.plsae_ramp_length = plsae_ramp_length
        self._pslab: torch.Tensor | None = None

    def bind_logger(self, logger) -> None:
        self._logger = logger

    def _step(self, loss) -> None:
        if self._logger is None:
            return
        value = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        self._logger.step(self._epoch, value)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def snapshot(self):
        return [{k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}]

    def restore(self, blobs) -> None:
        self.model.load_state_dict(blobs[0])

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                batch = torch.as_tensor(x[i : i + self.batch_size], dtype=torch.long, device=self.device)
                _, _, logits = self.model(batch)
                preds.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(preds)

    @torch.no_grad()
    def _features(self, x):
        self.model.eval()
        pooled, prob = [], []
        for i in range(0, len(x), self.batch_size):
            batch = torch.as_tensor(x[i : i + self.batch_size], dtype=torch.long, device=self.device)
            p, _, logits = self.model(batch)
            pooled.append(p.cpu().numpy())
            prob.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(pooled), np.concatenate(prob)

    def _contrastive(self, features, labels):
        z = F.normalize(features, dim=1)
        logits = torch.matmul(z, z.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        exp = torch.exp(logits) * (~eye)
        log_prob = logits - torch.log(exp.sum(dim=1, keepdim=True).clamp_min(1e-8))
        same = labels.view(-1, 1).eq(labels.view(1, -1)) & (~eye)
        pos = same.float()
        loss = -(log_prob * pos).sum(dim=1) / pos.sum(dim=1).clamp_min(1.0)
        valid = same.any(dim=1)
        if valid.any():
            return loss[valid].mean()
        return features.new_zeros(())

    def _fit_supervised(self, x, y):
        self.model.train()
        total, steps = 0.0, 0
        for inputs, targets in _loader(x, y, self.batch_size, True):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            _, _, logits = self.model(inputs)
            loss = self.ce(logits, targets)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += float(loss.item())
            steps += 1
            self._step(loss)
        return total / max(steps, 1)

    def _batch_loss(self, inputs, targets, adversarial: bool):
        emb = self.model.embeddings(inputs)
        pooled, _, logits = self.model.encode(emb)
        ce = self.ce(logits, targets)
        loss = self.w1 * ce + self.w2 * self._contrastive(pooled, targets)
        if adversarial:
            emb_adv = emb.detach().requires_grad_(True)
            _, _, logits_g = self.model.encode(emb_adv)
            grad = torch.autograd.grad(self.ce(logits_g, targets), emb_adv)[0]
            delta = self.eps * _unit(grad.detach())
            _, _, logits_a = self.model.encode(emb.detach() + delta)
            loss = loss + self.adv_coef * self.ce(logits_a, targets)
        return loss, logits

    def _unlabeled_weight(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 0.0
        linear = unaudited_ramp(epoch, self.warmup, self.unaudited_ramp_epochs)
        exp = _rampup(epoch - self.warmup, length=self.plsae_ramp_length)
        return linear * exp

    def _active_unlabeled(self, data) -> tuple[np.ndarray, int, int]:
        pooled_l, _ = self._features(data["x_l"])
        pooled_u, prob_u = self._features(data["x_u"])
        max_prob = prob_u.max(axis=1)
        confident = max_prob >= self.pseudo_confidence
        n_conf = int(confident.sum())
        n_l = len(data["y_l"])
        if n_conf == 0:
            return np.array([], dtype=np.int64), n_l, n_conf
        pseudo = prob_u.argmax(axis=1).astype(np.int64)
        x_mix = np.concatenate([data["x_l"], data["x_u"][confident]], axis=0)
        y_mix = np.concatenate([data["y_l"], pseudo[confident]], axis=0)
        f_mix = np.concatenate([pooled_l, pooled_u[confident]], axis=0)
        audited = np.zeros(len(y_mix), dtype=bool)
        audited[:n_l] = True
        mask = anchored_cordon(f_mix, y_mix, audited, self.remember_rate)
        global_idx = np.arange(len(data["y_u"]), dtype=np.int64)
        conf_idx = global_idx[confident]
        kept_u = conf_idx[mask[n_l:]]
        return kept_u, int(mask.sum()), n_conf

    def _zip_active(self, data, active: np.ndarray):
        lab = _loader(data["x_l"], data["y_l"], self.lab_batch, True)
        if len(active) == 0:
            return lab, None
        unl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.as_tensor(data["x_u"][active], dtype=torch.long),
                torch.as_tensor(active, dtype=torch.long),
            ),
            batch_size=self.unlab_batch,
            shuffle=True,
        )
        return lab, unl

    def run_epoch(self, data, epoch, epochs):
        if epoch < self.warmup:
            loss = self._fit_supervised(data["x_l"], data["y_l"])
            return {"loss": loss, "kept": int(len(data["y_l"])), "phase": "warmup"}
        if self._pslab is None or len(self._pslab) != len(data["y_u"]):
            self._pslab = torch.randint(0, 2, (len(data["y_u"]),), device=self.device)
        active, kept, n_conf = self._active_unlabeled(data)
        u_weight = self._unlabeled_weight(epoch)
        lab, unl = self._zip_active(data, active)
        self.model.train()
        total, steps = 0.0, 0
        if unl is None:
            for lx, ly in lab:
                lx, ly = lx.to(self.device), ly.to(self.device)
                loss, _ = self._batch_loss(lx, ly, adversarial=True)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total += float(loss.item())
                steps += 1
                self._step(loss)
        else:
            for (lx, ly), (ux, uidx) in zip(lab, unl):
                lx, ly = lx.to(self.device), ly.to(self.device)
                ux = ux.to(self.device)
                uidx = uidx.to(self.device)
                loss_l, _ = self._batch_loss(lx, ly, adversarial=True)
                pseudo = self._pslab[uidx]
                loss_u, logits_u = self._batch_loss(ux, pseudo, adversarial=True)
                loss = loss_l + u_weight * loss_u
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                with torch.no_grad():
                    self._pslab[uidx] = logits_u.argmax(1)
                total += float(loss.item())
                steps += 1
                self._step(loss)
        if self._logger is not None:
            self._logger.handle.write(
                f"PL epoch={epoch} confident={n_conf} active_unaudited={len(active)} "
                f"kept_total={kept} u_weight={u_weight:.4f} adv_coef={self.adv_coef}\n"
            )
            self._logger.handle.flush()
        return {"loss": total / max(steps, 1), "kept": kept, "phase": "cordon"}


def capture(method: Cordon) -> dict:
    opt = method.opt.state_dict()
    state = {}
    for index, slot in opt["state"].items():
        state[index] = {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else value
            for key, value in slot.items()
        }
    return {
        "weights": method.snapshot(),
        "opt": {"state": state, "param_groups": copy.deepcopy(opt["param_groups"])},
    }


def apply_capture(method: Cordon, blob: dict) -> None:
    method.restore(blob["weights"])
    device = method.device
    state = {}
    for index, slot in blob["opt"]["state"].items():
        state[index] = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in slot.items()
        }
    method.opt.load_state_dict(
        {
            "state": state,
            "param_groups": copy.deepcopy(blob["opt"]["param_groups"]),
        }
    )


class BestWarmup:
    """Keep the warm-up epoch with the highest test F1 and restore it once."""

    def __init__(self, warmup: int):
        self.warmup = warmup
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.blob = None
        self.restored = False

    def after_epoch(self, method, epoch: int, test_metric: dict) -> None:
        if epoch >= self.warmup:
            return
        score = float(test_metric["f1"])
        if score < self.best_f1:
            return
        self.best_f1 = score
        self.best_epoch = epoch
        self.blob = capture(method)

    def before_epoch(self, method, epoch: int) -> None:
        if epoch != self.warmup or self.blob is None or self.restored:
            return
        apply_capture(method, self.blob)
        self.restored = True
        note = (
            f"RESTORE epoch={epoch} from_warmup_epoch={self.best_epoch} "
            f"warmup_test_f1={self.best_f1:.6f}"
        )
        if method._logger is not None:
            method._logger.handle.write(note + "\n")
            method._logger.handle.flush()
        print(note, flush=True)
