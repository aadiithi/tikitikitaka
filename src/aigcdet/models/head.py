"""The trained part: a small MLP on top of frozen embeddings.

Deliberately small. With a 768-d frozen input and a few thousand training
images, a two-layer head with dropout is already at the point of diminishing
returns; anything larger memorises the training generators, which is the exact
failure we are trying to avoid. The whole model is ~200k trainable parameters
and trains in seconds on CPU.

Grouped validation: when the augmented feature bank contains four damaged views
of the same photograph, putting some views in train and others in validation
leaks. `train_head` splits on `source_index` (the original image), never on
rows, so a photo is wholly in one side or the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..utils.io import ensure_dir
from ..utils.logging import get_logger
from .calibration import TemperatureScaler, expected_calibration_error, pick_threshold

log = get_logger("models.head")


@dataclass
class HeadConfig:
    input_dim: int = 768
    hidden_dim: int = 256
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 256
    patience: int = 10
    val_frac: float = 0.15
    label_smoothing: float = 0.02
    seed: int = 1337


class DetectorHead(nn.Module):
    """Standardise -> Linear -> GELU -> Dropout -> Linear -> logit.

    The input standardisation lives inside the module as buffers so that the
    saved checkpoint is self-contained: `predict.py` loads one file and needs
    no separate scaler pickle.
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mu", torch.zeros(cfg.input_dim))
        self.register_buffer("sigma", torch.ones(cfg.input_dim))
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def set_standardisation(self, features: np.ndarray) -> None:
        mu = features.mean(axis=0)
        sigma = features.std(axis=0)
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.sigma.copy_(torch.as_tensor(np.clip(sigma, 1e-6, None), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mu) / self.sigma).squeeze(-1)

    @torch.no_grad()
    def logits(self, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        out = []
        for i in range(0, len(features), batch_size):
            x = torch.as_tensor(features[i : i + batch_size], dtype=torch.float32, device=device)
            out.append(self(x).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _grouped_split(
    source_index: np.ndarray,
    val_frac: float,
    seed: int,
    labels: np.ndarray | None = None,
    max_tries: int = 25,
):
    """Split row indices so that all rows sharing a source image go together.

    When labels are supplied we retry a few shuffles until the validation side
    contains both classes. A single-class validation split makes AUC undefined,
    which silently disables model selection - a failure that is easy to miss
    because training still "works" and simply produces a worse model.
    """
    groups = np.unique(source_index)
    for attempt in range(max_tries):
        rng = np.random.default_rng(seed + attempt)
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(val_frac * len(shuffled))))
        val_groups = set(shuffled[:n_val].tolist())
        is_val = np.array([g in val_groups for g in source_index])
        tr, va = np.where(~is_val)[0], np.where(is_val)[0]
        if labels is None:
            return tr, va
        if len(np.unique(labels[va])) > 1 and len(np.unique(labels[tr])) > 1:
            return tr, va
    log.warning(
        "could not find a grouped split with both classes on each side after %d tries - "
        "model selection will fall back to validation loss",
        max_tries,
    )
    return tr, va


def train_head(
    features: np.ndarray,
    labels: np.ndarray,
    source_index: np.ndarray | None = None,
    cfg: HeadConfig | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train the head, calibrate it, and choose an operating threshold.

    Returns a dict with the model, the fitted temperature, the chosen threshold
    and the validation history - everything `scripts/train_head.py` needs to
    write a checkpoint and a metrics JSON.
    """
    from sklearn.metrics import roc_auc_score

    cfg = cfg or HeadConfig(input_dim=features.shape[1])
    cfg.input_dim = features.shape[1]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    if source_index is None:
        source_index = np.arange(len(labels))

    tr_idx, va_idx = _grouped_split(source_index, cfg.val_frac, cfg.seed, labels=np.asarray(labels))
    log.info(
        "head training: %d train rows / %d val rows (%d source images held out)",
        len(tr_idx),
        len(va_idx),
        len(np.unique(source_index[va_idx])),
    )

    model = DetectorHead(cfg).to(device)
    model.set_standardisation(features[tr_idx])

    Xtr = torch.as_tensor(features[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.as_tensor(labels[tr_idx], dtype=torch.float32, device=device)
    Xva = features[va_idx]
    yva = labels[va_idx]

    # Class weighting: datasets are rarely balanced after a family-disjoint
    # split, and an unweighted loss quietly optimises for the majority class.
    n_pos = float((ytr == 1).sum().item())
    n_neg = float((ytr == 0).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    def _snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Seeded with the initial state so `best["state"]` is never None, even if
    # every epoch scores NaN. Selection uses AUC when it is defined and falls
    # back to negative validation loss when the split has a single class.
    best = {"score": -np.inf, "auc": float("nan"), "epoch": 0, "state": _snapshot()}
    history = []
    n = len(tr_idx)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(yva, dtype=torch.float32, device=device)

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            if cfg.label_smoothing:
                # Smoothing here is a robustness measure, not a regulariser
                # habit: some "real" images in public AIGC datasets are in fact
                # lightly edited, so hard 0/1 targets are over-claiming.
                yb = yb * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        sched.step()

        val_logits = model.logits(Xva)
        val_auc = float(roc_auc_score(yva, val_logits)) if len(np.unique(yva)) > 1 else float("nan")
        with torch.no_grad():
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(model(Xva_t), yva_t))
        history.append(
            {"epoch": epoch, "train_loss": total / max(n, 1), "val_auc": val_auc, "val_loss": val_loss}
        )

        selection = val_auc if not np.isnan(val_auc) else -val_loss
        if selection > best["score"]:
            best = {"score": selection, "auc": val_auc, "epoch": epoch, "state": _snapshot()}
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            log.info(
                "epoch %3d  loss %.4f  val_loss %.4f  val_auc %.4f",
                epoch, total / max(n, 1), val_loss, val_auc,
            )

        if epoch - best["epoch"] >= cfg.patience:
            log.info(
                "early stop at epoch %d (best epoch %d, val AUC %.4f)",
                epoch, best["epoch"], best["auc"],
            )
            break

    model.load_state_dict(best["state"])

    # Calibrate and pick the operating point on the validation split only.
    val_logits = model.logits(Xva)
    scaler = TemperatureScaler().fit(val_logits, yva)
    val_probs = scaler.transform(val_logits)
    threshold = pick_threshold(val_probs, yva, max_fpr=0.05)
    ece = expected_calibration_error(val_probs, yva)

    log.info(
        "best val AUC %.4f | temperature %.3f | ECE %.4f | threshold %.3f (FPR %.3f, TPR %.3f)",
        best["auc"], scaler.temperature, ece, threshold["threshold"], threshold["fpr"], threshold["tpr"],
    )

    return {
        "model": model,
        "config": cfg,
        "temperature": scaler.temperature,
        "threshold": threshold,
        "val_auc": best["auc"],
        "val_ece": ece,
        "best_epoch": best["epoch"],
        "history": history,
        "n_trainable_params": model.n_trainable(),
    }


def save_checkpoint(
    path: str | Path,
    model: DetectorHead,
    backbone_name: str,
    temperature: float,
    threshold: dict,
    extra: dict | None = None,
) -> Path:
    """Write one self-describing artefact: weights + how to reproduce inference."""
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "head_config": asdict(model.cfg),
            "backbone": backbone_name,
            "temperature": float(temperature),
            "threshold": threshold,
            "extra": extra or {},
            "format_version": 1,
        },
        p,
    )
    log.info("saved checkpoint -> %s", p)
    return p


def load_checkpoint(path: str | Path, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = HeadConfig(**ckpt["head_config"])
    model = DetectorHead(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model": model,
        "config": cfg,
        "backbone": ckpt.get("backbone", "clip-vit-l14"),
        "temperature": float(ckpt.get("temperature", 1.0)),
        "threshold": ckpt.get("threshold", {"threshold": 0.5}),
        "extra": ckpt.get("extra", {}),
    }
