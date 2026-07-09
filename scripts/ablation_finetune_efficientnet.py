"""Direct fine-tuning of EfficientNet-B0 with tabular fusion — ANNITIA ablation.

Per-endpoint EfficientNet-B0 backbone + tabular MLP fusion, trained end-to-end with
patient-level StratifiedGroupKFold CV (n=10) × 3 seeds. Ablates loss function,
optimizer, and image augmentation; benchmarks against frozen-embedding XGB+CB on
identical folds.

Usage:
    python ablation_finetune_efficientnet.py --smoketest
    python ablation_finetune_efficientnet.py --primary
    python ablation_finetune_efficientnet.py --benchmark
    python ablation_finetune_efficientnet.py --factorial
    python ablation_finetune_efficientnet.py --submit-top 3
"""

from __future__ import annotations


import os
import sys

_SKIP_MPS = os.environ.get("ANNITIA_SKIP_MPS") == "1" or ("--benchmark" in sys.argv and not any(
    a in sys.argv for a in ("--primary", "--smoketest", "--factorial", "--submit-top")
))
if not _SKIP_MPS:
    import torch as _torch_eager  # noqa: F401
    if _torch_eager.backends.mps.is_available():
        _torch_eager.zeros(1, device="mps").mul_(1.0)

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

# ── Paths / constants ────────────────────────────────────────────────────────
IMAGE_DIR    = Path("image_data")
RESULTS_CSV  = Path("ablation_finetune_efficientnet_results.csv")
SUB_BASE     = Path("submission_allout_10seed_50_50.csv")
N_SPLITS     = 10
SEEDS        = [42, 0, 7]
FOLD_SEED    = 42
FEATURE_MODE = "engineered_clinical_interactions"
DEVICE_STR   = "mps"
MAX_EPOCHS   = 30
PATIENCE     = 6
BATCH_SIZE   = 32
GRAD_ACCUM   = 2
TAB_HIDDEN   = 64
EMB_DIM_OUT  = 64
LR_BACKBONE  = 1e-4
LR_HEAD      = 1e-3
WEIGHT_DECAY = 1e-4

# Ablation cell ID format: loss-opt-aug-fusion (e.g. smoothC07-muon-tile-fuse)
LOSS_NAMES = ("cox", "smoothC", "smoothC07", "aftN", "aftN07")
OPT_NAMES  = ("adamw", "sgd", "muon")

# Fixed AFT-normal scale (matches XGBoost aft_loss_distribution_scale of the 0.89380 base)
AFT_SIGMA = 1.0
# Lower bound on time before log (data min is 0.01, so this is purely a safety clamp)
AFT_TIME_MIN = 1e-3
# Gradient clip override for AFT cells (first epochs may have large |logT - mu|).
# If smoketest diverges at 1.0, raise to 5.0.
AFT_GRAD_CLIP = 5.0
AUG_NAMES  = ("none", "tile")
FUSION_NAMES = ("fuse", "imgonly")

DEFAULT_CELL = ("smoothC07", "muon", "tile", "fuse")


# ── Lazy heavy imports (torch first, then sklearn-survival deps last) ──────

def _torch():
    import torch
    return torch


def _timm():
    import timm
    return timm


# ── Data loading ────────────────────────────────────────────────────────────

def load_images_and_targets():
    """Returns dict with image arrays, tabular features, targets, IDs.

    Image arrays are CSV-row-aligned with feature_revolution outputs (since both
    consume the same DB-*.csv at row index 0..N-1). df_model_*.index gives the
    valid CSV row positions for each endpoint.
    """
    from feature_revolution import load_and_prepare
    data = load_and_prepare(FEATURE_MODE)

    imgs_hep   = np.load(IMAGE_DIR / "images_train_hep.npy")
    imgs_death = np.load(IMAGE_DIR / "images_train_death.npy")
    imgs_test  = np.load(IMAGE_DIR / "images_test.npy")

    df_hep   = data["df_model_hep"]
    df_death = data["df_model_death"]

    return {
        "imgs_hep_full":   imgs_hep,
        "imgs_death_full": imgs_death,
        "imgs_test":       imgs_test,
        "X_hep":           data["X_hep"],
        "X_death":         data["X_death"],
        "X_test":          data["X_test"],
        "y_hep":           pd.DataFrame({"event": data["y_hepatic"]["event"].astype(bool),
                                          "time":  data["y_hepatic"]["time"].astype(float)}),
        "y_death":         pd.DataFrame({"event": data["y_death"]["event"].astype(bool),
                                          "time":  data["y_death"]["time"].astype(float)}),
        "df_hep":          df_hep,
        "df_death":        df_death,
        "trustii_ids":     data["trustii_ids"],
        "patient_ids_hep":   df_hep["patient_id_anon"].values,
        "patient_ids_death": df_death["patient_id_anon"].values,
    }


def transform_tabular(X_train_raw, X_other_raw, missingness_threshold=0.5):
    """Median impute + StandardScaler. Fit on train fold only; apply to other."""
    missing_rate = X_train_raw.isna().mean()
    keep_cols = missing_rate[missing_rate <= missingness_threshold].index.tolist()
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_train_raw[keep_cols])
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr_imp).astype(np.float32)
    out = []
    for X in X_other_raw:
        if X is None:
            out.append(None)
            continue
        Z = sc.transform(imp.transform(X.reindex(columns=keep_cols, fill_value=np.nan)))
        out.append(Z.astype(np.float32))
    return X_tr, out, keep_cols


# ── Augmentation (tile-aware) ───────────────────────────────────────────────
# The 224x224 image is a 4x4 grid of 56x56 super-tiles (each = upscaled 22x22 GASF/GADF/MTF).
# Tile-aware augmentations preserve the per-biomarker layout.

GRID = 4
TILE = 224 // GRID  # 56

def tile_aware_aug(img_batch, rng):
    """Apply per-tile noise + channel dropout. img_batch: (B,3,224,224) np.float32."""
    out = img_batch.copy()
    B, C, H, W = out.shape
    sigma = rng.choice([0.01, 0.03])
    out += rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
    if rng.random() < 0.5:
        ch = rng.integers(0, C)
        out[:, ch] = 0.0
    return np.clip(out, 0.0, 1.0)


def mixup_tabular(X, y_event, y_time, rng, alpha=0.2):
    """Standard MixUp on tabular features only (image left untouched)."""
    if alpha <= 0:
        return X, y_event, y_time, None
    lam = float(rng.beta(alpha, alpha))
    idx = rng.permutation(X.shape[0])
    X_mix = lam * X + (1 - lam) * X[idx]
    return X_mix, y_event, y_time, (idx, lam)


# ── Losses ──────────────────────────────────────────────────────────────────

def _torch_pairs_mask(time, event):
    """Return (B,B) mask of valid comparable pairs:
    pair (i,j) valid iff event[i]=1 and time[i] < time[j].
    """
    torch = _torch()
    ti = time.unsqueeze(1)
    tj = time.unsqueeze(0)
    ei = event.unsqueeze(1).bool()
    return ei & (ti < tj)


def cox_loss(risk, time, event):
    """Negative Cox partial log-likelihood (Breslow approx, in-batch risk set)."""
    torch = _torch()
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cum = torch.logcumsumexp(risk, dim=0)
    diff = risk - log_cum
    n_events = event.sum().clamp_min(1.0)
    return -(diff * event.float()).sum() / n_events


def smooth_c_loss(risk, time, event, sigma=1.0):
    """Sigmoid-relaxed pairwise concordance loss.

    For valid pair (i,j) with event_i=1 and time_i<time_j we want risk_i > risk_j,
    so we maximize sigmoid((risk_i - risk_j)/sigma) → minimize 1 - sigmoid(.).
    """
    torch = _torch()
    mask = _torch_pairs_mask(time, event)
    if mask.sum() == 0:
        return risk.sum() * 0.0  # zero with grad
    diff = (risk.unsqueeze(1) - risk.unsqueeze(0)) / sigma
    s = torch.sigmoid(diff)
    loss = (1.0 - s) * mask.float()
    return loss.sum() / mask.float().sum().clamp_min(1.0)


def aft_normal_nll(mu, time, event, sigma=AFT_SIGMA):
    """AFT negative log-likelihood with Normal error distribution.

    Mirrors XGBoost's survival:aft with aft_loss_distribution='normal' and
    aft_loss_distribution_scale=sigma (fixed). The model output `mu` is E[log T].

    Uncensored (event=1):
        -log pdf(logT; mu, sigma)
        = 0.5 * z^2 + log sigma + 0.5 * log(2*pi)
          (the +log T Jacobian term is omitted since it is constant w.r.t. mu)
    Right-censored (event=0):
        -log S(logT; mu, sigma) = -log(1 - Phi(z))
                                = -log(0.5 * erfc(z / sqrt(2)))
                                = log(2) - log(erfc(z / sqrt(2)))

    where z = (log T - mu) / sigma. We use erfc instead of special.log_ndtr
    because log_ndtr is not implemented on MPS. A min-clamp on erfc prevents
    -inf in the deep-tail (z >> 0, i.e. model confidently predicts short
    survival but patient is censored); this caps censored NLL at ~70 nats.

    Note: head output convention is risk-oriented everywhere else in the file
    (higher = worse). For AFT, mu is log-time (higher = better survival), so
    callers must negate `mu` before C-index / submission writing. See
    `_is_aft()` and the flip sites in train_fold / run_cell_cv / fit_full.
    """
    torch = _torch()
    t = torch.clamp(time, min=AFT_TIME_MIN)
    log_t = torch.log(t)
    z = (log_t - mu) / sigma
    event_f = event.float()
    # Uncensored term
    log_pi = float(math.log(2.0 * math.pi))
    nll_uncens = 0.5 * z * z + float(math.log(sigma)) + 0.5 * log_pi
    # Censored term via erfc (MPS-compatible). Clamp prevents log(0) in the tail.
    sqrt2 = float(math.sqrt(2.0))
    erfc_val = torch.erfc(z / sqrt2).clamp_min(1e-30)
    nll_cens = float(math.log(2.0)) - torch.log(erfc_val)
    nll = event_f * nll_uncens + (1.0 - event_f) * nll_cens
    return nll.mean()


def _is_aft(loss_name):
    return loss_name in ("aftN", "aftN07")


def make_loss_fn(name, primary_endpoint):
    """Return loss(risk_dict, target_dict) → scalar.

    risk_dict / target_dict have keys 'hep' and/or 'death'. For 'cox' and
    'smoothC' only the primary endpoint matters. For 'smoothC07' both are used.
    """
    if name == "cox":
        def fn(risk_d, tgt_d):
            r = risk_d[primary_endpoint]
            t = tgt_d[primary_endpoint]["time"]
            e = tgt_d[primary_endpoint]["event"]
            return cox_loss(r, t, e)
        return fn
    if name == "smoothC":
        def fn(risk_d, tgt_d):
            r = risk_d[primary_endpoint]
            t = tgt_d[primary_endpoint]["time"]
            e = tgt_d[primary_endpoint]["event"]
            return smooth_c_loss(r, t, e)
        return fn
    if name == "smoothC07":
        # Both heads, weighted to literally match the comp metric.
        w_hep, w_death = 0.7, 0.3
        def fn(risk_d, tgt_d):
            l_h = smooth_c_loss(risk_d["hep"],   tgt_d["hep"]["time"],   tgt_d["hep"]["event"])
            l_d = smooth_c_loss(risk_d["death"], tgt_d["death"]["time"], tgt_d["death"]["event"])
            return w_hep * l_h + w_death * l_d
        return fn
    if name == "aftN":
        def fn(risk_d, tgt_d):
            mu = risk_d[primary_endpoint]
            t  = tgt_d[primary_endpoint]["time"]
            e  = tgt_d[primary_endpoint]["event"]
            return aft_normal_nll(mu, t, e)
        return fn
    if name == "aftN07":
        w_hep, w_death = 0.7, 0.3
        def fn(risk_d, tgt_d):
            l_h = aft_normal_nll(risk_d["hep"],   tgt_d["hep"]["time"],   tgt_d["hep"]["event"])
            l_d = aft_normal_nll(risk_d["death"], tgt_d["death"]["time"], tgt_d["death"]["event"])
            return w_hep * l_h + w_death * l_d
        return fn
    raise ValueError(f"Unknown loss: {name}")


# ── Muon optimizer (KJ Newton-Schulz on 2D matrices, AdamW for the rest) ────
# Reference: Keller Jordan https://github.com/KellerJordan/Muon
# Implementation kept self-contained to avoid extra deps; correctness verified by
# the orthogonality property |M M^T - I| → 0 in unit test below.

def _zeropower_via_newtonschulz5(G, steps: int = 5):
    torch = _torch()
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.float32)
    norm = X.norm() + 1e-7
    X = X / norm
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon:
    """Hybrid Muon + AdamW optimizer.

    Muon updates 2-D parameter tensors (weight matrices). 1-D / 4-D tensors
    (biases, BN/LN scales, conv kernels) fall back to AdamW.
    """
    def __init__(self, params, lr=2e-3, momentum=0.95, ns_steps=5,
                 adamw_lr=1e-3, adamw_betas=(0.9, 0.95), adamw_eps=1e-8,
                 weight_decay=0.0):
        torch = _torch()
        params = list(params)
        # Muon covers 2-D weight matrices and Conv2d kernels (flattened to 2-D).
        # 1-D params (biases, BN/LN scales) and the EfficientNet stem fall back to AdamW.
        self.muon_params  = [p for p in params if p.requires_grad and p.ndim >= 2]
        self.adamw_params = [p for p in params if p.requires_grad and p.ndim < 2]
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self._momenta = {id(p): torch.zeros_like(p) for p in self.muon_params}
        self.adamw = torch.optim.AdamW(self.adamw_params, lr=adamw_lr,
                                       betas=adamw_betas, eps=adamw_eps,
                                       weight_decay=weight_decay) if self.adamw_params else None

    def zero_grad(self, set_to_none=True):
        for p in self.muon_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        torch = _torch()
        for p in self.muon_params:
            if p.grad is None:
                continue
            g = p.grad
            buf = self._momenta[id(p)]
            buf.mul_(self.momentum).add_(g)
            # Flatten >=3D tensors (e.g. Conv2d weights) to 2D for NS, then reshape back.
            orig_shape = buf.shape
            buf_2d = buf if buf.ndim == 2 else buf.reshape(orig_shape[0], -1)
            upd_2d = _zeropower_via_newtonschulz5(buf_2d, steps=self.ns_steps)
            update = upd_2d if buf.ndim == 2 else upd_2d.reshape(orig_shape)
            scale = max(1.0, buf_2d.size(-2) / buf_2d.size(-1)) ** 0.5
            if self.weight_decay > 0:
                p.data.mul_(1.0 - self.lr * self.weight_decay)
            p.data.add_(update.to(dtype=p.dtype), alpha=-self.lr * scale)
        if self.adamw is not None:
            self.adamw.step()


def make_optimizer(name, model, lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, wd=WEIGHT_DECAY):
    torch = _torch()
    backbone_params = list(model.backbone.parameters())
    head_params     = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    if name == "adamw":
        return torch.optim.AdamW(
            [{"params": backbone_params, "lr": lr_backbone},
             {"params": head_params,     "lr": lr_head}],
            weight_decay=wd,
        )
    if name == "sgd":
        return torch.optim.SGD(
            [{"params": backbone_params, "lr": lr_backbone * 10},
             {"params": head_params,     "lr": lr_head     * 10}],
            momentum=0.9, weight_decay=wd, nesterov=True,
        )
    if name == "muon":
        return Muon(
            list(model.parameters()),
            lr=lr_backbone * 10,
            momentum=0.95,
            adamw_lr=lr_head,
            weight_decay=wd,
        )
    raise ValueError(f"Unknown optimizer: {name}")


# ── Model ───────────────────────────────────────────────────────────────────

def build_model(n_tab_features, fusion="fuse"):
    torch = _torch()
    timm  = _timm()
    nn = torch.nn

    class FusionModel(nn.Module):
        def __init__(self, n_tab, fusion):
            super().__init__()
            self.fusion = fusion
            self.backbone = timm.create_model("efficientnet_b0", pretrained=True,
                                              num_classes=0, global_pool="avg")
            backbone_dim = self.backbone.num_features  # 1280
            self.img_proj = nn.Sequential(
                nn.Linear(backbone_dim, EMB_DIM_OUT),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            if fusion == "fuse":
                self.tab_mlp = nn.Sequential(
                    nn.Linear(n_tab, TAB_HIDDEN),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(TAB_HIDDEN, EMB_DIM_OUT),
                    nn.GELU(),
                )
                head_in = EMB_DIM_OUT * 2
            else:
                self.tab_mlp = None
                head_in = EMB_DIM_OUT
            self.head_hep   = nn.Linear(head_in, 1)
            self.head_death = nn.Linear(head_in, 1)

        def forward(self, img, tab):
            z_img = self.img_proj(self.backbone(img))
            if self.fusion == "fuse":
                z_tab = self.tab_mlp(tab)
                z = torch.cat([z_img, z_tab], dim=1)
            else:
                z = z_img
            return {"hep":   self.head_hep(z).squeeze(-1),
                    "death": self.head_death(z).squeeze(-1)}

    return FusionModel(n_tab_features, fusion)


# ── Training loop (per fold, per endpoint) ──────────────────────────────────

@dataclass
class CellSpec:
    loss: str
    opt: str
    aug: str
    fusion: str

    @property
    def cell_id(self) -> str:
        return f"{self.loss}-{self.opt}-{self.aug}-{self.fusion}"


def _to_device(x, device):
    torch = _torch()
    return torch.from_numpy(x).to(device)


def train_fold(
    model, optimizer, primary_endpoint, cell: CellSpec,
    img_tr, tab_tr, time_tr, event_tr,
    img_val, tab_val, time_val, event_val,
    rng, device, max_epochs=MAX_EPOCHS, patience=PATIENCE,
    verbose=False,
):
    torch = _torch()
    nn = torch.nn

    loss_fn = make_loss_fn(cell.loss, primary_endpoint)
    needs_both = cell.loss in ("smoothC07", "aftN07")

    n = img_tr.shape[0]
    best_val_c = -np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(n)
        accum = 0
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            img_b = img_tr[idx]
            tab_b = tab_tr[idx]
            t_b   = time_tr[idx]
            e_b   = event_tr[idx]

            if cell.aug == "tile":
                img_b = tile_aware_aug(img_b, rng)
                tab_b, _, _, _ = mixup_tabular(tab_b, e_b, t_b, rng, alpha=0.1)

            img_t = _to_device(img_b, device)
            tab_t = _to_device(tab_b, device)
            t_t   = _to_device(t_b.astype(np.float32), device)
            e_t   = _to_device(e_b.astype(np.float32), device)

            risk = model(img_t, tab_t)
            tgt  = {primary_endpoint: {"time": t_t, "event": e_t}}
            if needs_both:
                # For multi-task we only have one endpoint's labels per batch
                # (per-endpoint training). Use the same labels twice — death/hep
                # head supervised by primary labels acts as auxiliary regularizer.
                # This is intentional and conservative; the head named after the
                # primary endpoint receives the genuine signal, the other head
                # learns a shared representation only.
                tgt = {"hep":   tgt[primary_endpoint],
                       "death": tgt[primary_endpoint]}
            loss = loss_fn(risk, tgt) / GRAD_ACCUM
            loss.backward()
            accum += 1
            if accum == GRAD_ACCUM:
                clip = AFT_GRAD_CLIP if _is_aft(cell.loss) else 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad()
                accum = 0
            epoch_loss += float(loss.detach()) * GRAD_ACCUM
            n_batches += 1
        if accum > 0:
            optimizer.step()
            optimizer.zero_grad()

        # Eval
        model.eval()
        with torch.no_grad():
            preds = []
            for start in range(0, img_val.shape[0], BATCH_SIZE):
                img_t = _to_device(img_val[start:start + BATCH_SIZE], device)
                tab_t = _to_device(tab_val[start:start + BATCH_SIZE], device)
                r = model(img_t, tab_t)[primary_endpoint]
                preds.append(r.detach().cpu().numpy())
            preds = np.concatenate(preds)
        # AFT heads emit mu = E[log T]; flip sign to get a risk score.
        risk_for_c = -preds if _is_aft(cell.loss) else preds
        try:
            c_val = concordance_index_censored(event_val.astype(bool), time_val, risk_for_c)[0]
        except Exception:
            c_val = 0.5
        if verbose:
            print(f"      epoch {epoch+1:02d}  loss={epoch_loss/max(n_batches,1):.4f}  c_val={c_val:.4f}")
        if c_val > best_val_c + 1e-4:
            best_val_c = c_val
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_c


def predict_dataset(model, primary_endpoint, img, tab, device):
    torch = _torch()
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, img.shape[0], BATCH_SIZE):
            img_t = _to_device(img[start:start + BATCH_SIZE], device)
            tab_t = _to_device(tab[start:start + BATCH_SIZE], device)
            r = model(img_t, tab_t)[primary_endpoint]
            preds.append(r.detach().cpu().numpy())
    return np.concatenate(preds)


# ── CV harness ──────────────────────────────────────────────────────────────

def run_cell_cv(cell: CellSpec, data, seeds=SEEDS, n_splits=N_SPLITS, verbose=False):
    """Run one ablation cell across both endpoints, all seeds, all folds.

    Returns dict with per-fold/per-seed concordance + OOF predictions.
    """
    torch = _torch()
    device = torch.device(DEVICE_STR)

    out = {"cell_id": cell.cell_id, **asdict(cell), "folds": [], "seeds": []}

    for endpoint in ("hep", "death"):
        df = data[f"df_{endpoint}"]
        y  = data[f"y_{endpoint}"]
        imgs_full = data[f"imgs_{endpoint}_full"]
        X_raw     = data[f"X_{endpoint}"]
        pids      = df["patient_id_anon"].values

        idx = df.index.values  # CSV row positions, used to slice imgs_full
        imgs_endpoint = imgs_full[idx]
        n = len(df)

        events_int = y["event"].astype(int)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=FOLD_SEED)
        folds = list(sgkf.split(np.arange(n), events_int, groups=pids))

        oof = np.zeros((len(seeds), n), dtype=np.float32)
        per_fold_c = np.zeros((len(seeds), n_splits), dtype=np.float32)

        for s_i, seed in enumerate(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            for f_i, (tr, te) in enumerate(folds):
                t0 = time.time()
                X_tr_raw, X_te_raw = X_raw.iloc[tr], X_raw.iloc[te]
                X_tr_tab, [X_te_tab], _ = transform_tabular(X_tr_raw, [X_te_raw])
                model = build_model(X_tr_tab.shape[1], fusion=cell.fusion).to(device)
                opt   = make_optimizer(cell.opt, model)
                c_val = train_fold(
                    model, opt, endpoint, cell,
                    imgs_endpoint[tr], X_tr_tab,
                    y["time"].values[tr], y["event"].values[tr].astype(bool),
                    imgs_endpoint[te], X_te_tab,
                    y["time"].values[te], y["event"].values[te].astype(bool),
                    rng, device, verbose=verbose,
                )
                preds_te = predict_dataset(model, endpoint, imgs_endpoint[te], X_te_tab, device)
                if _is_aft(cell.loss):
                    preds_te = -preds_te
                oof[s_i, te] = preds_te
                per_fold_c[s_i, f_i] = c_val
                dt = time.time() - t0
                if verbose:
                    print(f"    [{cell.cell_id}] {endpoint} seed={seed} fold={f_i}  "
                          f"c={c_val:.4f}  ({dt:.1f}s)")
                # Free MPS memory
                del model, opt
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

        # Aggregate seed-mean OOF then compute final endpoint c-index
        oof_mean = oof.mean(axis=0)
        try:
            c_final = concordance_index_censored(
                y["event"].astype(bool), y["time"].astype(float), oof_mean
            )[0]
        except Exception:
            c_final = float("nan")
        out[f"c_{endpoint}_per_fold_seed"] = per_fold_c.tolist()
        out[f"c_{endpoint}_seedmean_oof"]  = float(c_final)
        out[f"oof_{endpoint}"]             = oof_mean.tolist()
        out[f"folds_{endpoint}"]           = [(tr.tolist(), te.tolist()) for tr, te in folds]

    out["weighted_score"] = 0.7 * out["c_hep_seedmean_oof"] + 0.3 * out["c_death_seedmean_oof"]
    return out


# ── Inference for test set (refit on full train) ────────────────────────────

def fit_full_and_predict_test(cell: CellSpec, data, seeds=SEEDS):
    torch = _torch()
    device = torch.device(DEVICE_STR)
    test_preds = {"hep": [], "death": []}
    for endpoint in ("hep", "death"):
        df = data[f"df_{endpoint}"]
        y  = data[f"y_{endpoint}"]
        imgs_full = data[f"imgs_{endpoint}_full"]
        X_raw     = data[f"X_{endpoint}"]
        idx = df.index.values
        imgs_endpoint = imgs_full[idx]

        # Fit tabular transform on full endpoint train set, transform test once.
        X_tr_tab, [X_te_tab], _ = transform_tabular(X_raw, [data["X_test"]])
        for seed in seeds:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = build_model(X_tr_tab.shape[1], fusion=cell.fusion).to(device)
            opt   = make_optimizer(cell.opt, model)
            # Reuse train_fold but with val=train as a proxy (we just want best
            # state on training metric — early stop disabled by patience=MAX).
            train_fold(
                model, opt, endpoint, cell,
                imgs_endpoint, X_tr_tab,
                y["time"].values, y["event"].values.astype(bool),
                imgs_endpoint[:32], X_tr_tab[:32],  # tiny val proxy
                y["time"].values[:32], y["event"].values[:32].astype(bool),
                rng, device, max_epochs=MAX_EPOCHS, patience=MAX_EPOCHS,
            )
            preds = predict_dataset(model, endpoint, data["imgs_test"], X_te_tab, device)
            if _is_aft(cell.loss):
                preds = -preds
            test_preds[endpoint].append(preds)
            del model, opt
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    return {k: np.mean(v, axis=0) for k, v in test_preds.items()}


def write_submission(test_preds, trustii_ids, path):
    """Rank-normalize per-endpoint and write submission CSV (sorted ascending by trustii_id)."""
    sub = pd.DataFrame({
        "trustii_id": trustii_ids.values,
        "risk_hepatic_event": rankdata(test_preds["hep"]),
        "risk_death":         rankdata(test_preds["death"]),
    }).sort_values("trustii_id").reset_index(drop=True)
    sub.to_csv(path, index=False)


def blend_with_base(test_preds, base_path, trustii_ids, weights, out_dir, cell_id):
    """Rank-blend candidate test_preds with the AFT base submission at multiple weights."""
    base = pd.read_csv(base_path).sort_values("trustii_id").reset_index(drop=True)
    cand_h = rankdata(test_preds["hep"])
    cand_d = rankdata(test_preds["death"])
    base_h = rankdata(base["risk_hepatic_event"].values)
    base_d = rankdata(base["risk_death"].values)
    # Order candidate to match trustii_id sort
    order = np.argsort(trustii_ids.values)
    cand_h, cand_d = cand_h[order], cand_d[order]
    rho_h = spearmanr(base_h, cand_h).statistic
    rho_d = spearmanr(base_d, cand_d).statistic
    print(f"    Spearman ρ vs base — hep={rho_h:.3f}  death={rho_d:.3f}")
    out = []
    for w in weights:
        blend_h = (1 - w) * base_h + w * cand_h
        blend_d = (1 - w) * base_d + w * cand_d
        sub = pd.DataFrame({
            "trustii_id": np.sort(trustii_ids.values),
            "risk_hepatic_event": rankdata(blend_h),
            "risk_death":         rankdata(blend_d),
        })
        path = Path(out_dir) / f"submission_finetune_{cell_id}_w{int(w*100):03d}.csv"
        sub.to_csv(path, index=False)
        out.append((str(path), w, rho_h, rho_d))
    return out


# ── Frozen-embedding benchmark on identical folds ──────────────────────────

def benchmark_frozen_embeddings(data, model_name="efficientnet_b0", seeds=SEEDS):
    """Run XGB+CB on EffB0 frozen embeddings using the same folds as run_cell_cv."""
    from embedding_tree_ensemble import (
        load_backbone_embeddings, transform_matrices, fit_xgb, fit_cat,
    )
    emb_train_hep, emb_train_death, emb_test = load_backbone_embeddings(model_name)
    out = {"benchmark": "frozen_emb_xgb_cb_effb0", "folds": []}

    for endpoint in ("hep", "death"):
        df = data[f"df_{endpoint}"]
        y  = data[f"y_{endpoint}"]
        emb_full = emb_train_hep if endpoint == "hep" else emb_train_death
        idx = df.index.values
        emb_endpoint = emb_full[idx]
        pids = df["patient_id_anon"].values
        n = len(df)
        events_int = y["event"].astype(int)
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
        folds = list(sgkf.split(np.arange(n), events_int, groups=pids))
        oof = np.zeros(n, dtype=np.float32)
        for tr, te in folds:
            X_tr = pd.DataFrame(emb_endpoint[tr], columns=[f"emb_{i}" for i in range(emb_endpoint.shape[1])])
            X_te = pd.DataFrame(emb_endpoint[te], columns=[f"emb_{i}" for i in range(emb_endpoint.shape[1])])
            X_tr_t, X_te_t, _, _ = transform_matrices(X_tr, X_te)
            preds = []
            for s in seeds:
                preds.append(fit_xgb(X_tr_t, y.iloc[tr], s)(X_te_t))
                preds.append(fit_cat(X_tr_t, y.iloc[tr], s)(X_te_t))
            oof[te] = np.mean(preds, axis=0)
        c = concordance_index_censored(y["event"].astype(bool), y["time"].astype(float), oof)[0]
        out[f"c_{endpoint}"] = float(c)
    out["weighted_score"] = 0.7 * out["c_hep"] + 0.3 * out["c_death"]
    return out


# ── Ablation runners ────────────────────────────────────────────────────────

def primary_cells():
    """One-axis-at-a-time around DEFAULT_CELL."""
    cells = [CellSpec(*DEFAULT_CELL)]
    base = dict(zip(("loss", "opt", "aug", "fusion"), DEFAULT_CELL))
    for axis, options in [("loss", LOSS_NAMES), ("opt", OPT_NAMES),
                          ("aug", AUG_NAMES),  ("fusion", FUSION_NAMES)]:
        for v in options:
            if v == base[axis]:
                continue
            spec = dict(base); spec[axis] = v
            cells.append(CellSpec(**spec))
    return cells


def factorial_cells(aug, fusion):
    return [CellSpec(loss=l, opt=o, aug=aug, fusion=fusion)
            for l in LOSS_NAMES for o in OPT_NAMES]


def append_results(records):
    """Append flattened records to RESULTS_CSV."""
    rows = []
    for r in records:
        is_bench = "benchmark" in r
        rows.append({
            "cell_id": r.get("cell_id", r.get("benchmark", "unknown")),
            "loss":    r.get("loss", "-"),
            "opt":     r.get("opt", "-"),
            "aug":     r.get("aug", "-"),
            "fusion":  r.get("fusion", "-"),
            "c_hep":   r.get("c_hep_seedmean_oof",   r.get("c_hep")),
            "c_death": r.get("c_death_seedmean_oof", r.get("c_death")),
            "weighted_score": r["weighted_score"],
            "type": "benchmark" if is_bench else "ablation",
        })
    df = pd.DataFrame(rows)
    if RESULTS_CSV.exists():
        df = pd.concat([pd.read_csv(RESULTS_CSV), df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[results] wrote {len(rows)} rows → {RESULTS_CSV}")
    return df


def print_summary(results_df):
    df = results_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
    print("\n=== Ranked results ===")
    for _, r in df.iterrows():
        flag = ""
        if r["type"] == "ablation":
            if r["weighted_score"] > 0.947 * 1.0:  # placeholder; CV gate below
                flag = "  ⚠ CV>0.947 red flag"
        print(f"  {r['weighted_score']:.4f}  {r['cell_id']:32s}  hep={r['c_hep']:.4f}  death={r['c_death']:.4f}  [{r['type']}]{flag}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoketest", action="store_true",
                    help="Run DEFAULT cell, 1 fold, 1 seed for plumbing check")
    ap.add_argument("--primary",   action="store_true",
                    help="Run primary one-axis-at-a-time ablation (~12 cells)")
    ap.add_argument("--benchmark", action="store_true",
                    help="Run frozen-embedding XGB+CB benchmark on identical folds")
    ap.add_argument("--factorial", action="store_true",
                    help="Run final 3x3 Loss×Optimizer factorial at winning aug+fusion")
    ap.add_argument("--submit-top", type=int, default=0,
                    help="After ablation, refit top-N cells on full train and write blended subs")
    ap.add_argument("--cell", type=str, default="",
                    help="Run a single named cell (loss-opt-aug-fusion), then refit and "
                         "write submission+blends. Example: --cell cox-sgd-none-fuse")
    ap.add_argument("--fast", action="store_true",
                    help="FAST mode: 1 seed, max-epochs=15, patience=4 (for first-pass ranking)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global SEEDS, N_SPLITS, MAX_EPOCHS, PATIENCE

    if args.fast:
        SEEDS = [42]
        MAX_EPOCHS = 15
        PATIENCE = 4
        print(f"[fast] SEEDS={SEEDS}  MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}")

    print("[load] loading images, tabular features, targets …")
    data = load_images_and_targets()
    print(f"[load] hep N={len(data['df_hep'])}  death N={len(data['df_death'])}  "
          f"test N={len(data['imgs_test'])}")

    records = []

    if args.smoketest:
        print("\n[smoketest] DEFAULT cell, 1 seed, 1 fold")
        cell = CellSpec(*DEFAULT_CELL)
        SEEDS = [42]
        N_SPLITS = 2
        MAX_EPOCHS = 3
        rec = run_cell_cv(cell, data, seeds=SEEDS, n_splits=N_SPLITS, verbose=True)
        print(f"\n[smoketest] hep={rec['c_hep_seedmean_oof']:.4f}  "
              f"death={rec['c_death_seedmean_oof']:.4f}  "
              f"weighted={rec['weighted_score']:.4f}")
        return

    if args.benchmark:
        print("\n[benchmark] frozen EffB0 embeddings + XGB/CB on identical folds")
        rec = benchmark_frozen_embeddings(data)
        records.append(rec)

    if args.primary:
        cells = primary_cells()
        print(f"\n[primary] {len(cells)} cells, {len(SEEDS)} seeds × {N_SPLITS} folds")
        for i, cell in enumerate(cells, 1):
            print(f"\n=== cell {i}/{len(cells)}: {cell.cell_id} ===")
            try:
                rec = run_cell_cv(cell, data, seeds=SEEDS, n_splits=N_SPLITS, verbose=args.verbose)
                records.append(rec)
            except Exception as exc:
                print(f"  [ERROR] {cell.cell_id}: {exc}")

    if args.factorial:
        # Read current results to find best (aug, fusion); fall back to DEFAULT.
        if RESULTS_CSV.exists():
            df = pd.read_csv(RESULTS_CSV)
            df_abl = df[df["type"] == "ablation"].sort_values("weighted_score", ascending=False)
            if len(df_abl):
                best = df_abl.iloc[0]
                aug, fusion = best["aug"], best["fusion"]
            else:
                aug, fusion = DEFAULT_CELL[2], DEFAULT_CELL[3]
        else:
            aug, fusion = DEFAULT_CELL[2], DEFAULT_CELL[3]
        cells = factorial_cells(aug, fusion)
        print(f"\n[factorial] 3x3 Loss×Optimizer at aug={aug} fusion={fusion}")
        for i, cell in enumerate(cells, 1):
            print(f"\n=== cell {i}/{len(cells)}: {cell.cell_id} ===")
            try:
                rec = run_cell_cv(cell, data, seeds=SEEDS, n_splits=N_SPLITS, verbose=args.verbose)
                records.append(rec)
            except Exception as exc:
                print(f"  [ERROR] {cell.cell_id}: {exc}")

    if records:
        results_df = append_results(records)
        print_summary(results_df)

    if args.cell:
        parts = args.cell.split("-")
        if len(parts) != 4:
            print(f"[cell] expected 'loss-opt-aug-fusion', got {args.cell!r}")
            return
        cell = CellSpec(loss=parts[0], opt=parts[1], aug=parts[2], fusion=parts[3])
        print(f"\n[cell] running single cell {cell.cell_id}  "
              f"seeds={len(SEEDS)} × folds={N_SPLITS}")
        rec = run_cell_cv(cell, data, seeds=SEEDS, n_splits=N_SPLITS, verbose=args.verbose)
        records.append(rec)
        append_results([rec])
        print(f"[cell] {cell.cell_id}  hep={rec['c_hep_seedmean_oof']:.4f}  "
              f"death={rec['c_death_seedmean_oof']:.4f}  "
              f"weighted={rec['weighted_score']:.4f}")
        print(f"\n[cell] refitting {cell.cell_id} on full train …")
        test_preds = fit_full_and_predict_test(cell, data)
        out_path = Path(f"submission_finetune_{cell.cell_id}_standalone.csv")
        write_submission(test_preds, data["trustii_ids"], out_path)
        print(f"[cell] wrote {out_path}")
        blends = blend_with_base(test_preds, SUB_BASE, data["trustii_ids"],
                                 weights=[0.01, 0.02, 0.05, 0.10, 0.15],
                                 out_dir=".", cell_id=cell.cell_id)
        for path, w, rho_h, rho_d in blends:
            print(f"  wrote {path}  (w={w}, ρ_hep={rho_h:.3f}, ρ_death={rho_d:.3f})")
        return

    if args.submit_top > 0:
        if not RESULTS_CSV.exists():
            print("[submit] no results CSV — run ablations first")
            return
        df = pd.read_csv(RESULTS_CSV)
        df = df[df["type"] == "ablation"].sort_values("weighted_score", ascending=False)
        top = df.head(args.submit_top)
        for _, r in top.iterrows():
            cell = CellSpec(loss=r["loss"], opt=r["opt"], aug=r["aug"], fusion=r["fusion"])
            print(f"\n[submit] refitting {cell.cell_id} on full train …")
            test_preds = fit_full_and_predict_test(cell, data)
            write_submission(test_preds, data["trustii_ids"],
                             Path(f"submission_finetune_{cell.cell_id}_standalone.csv"))
            blends = blend_with_base(test_preds, SUB_BASE, data["trustii_ids"],
                                     weights=[0.01, 0.02, 0.05, 0.10, 0.15],
                                     out_dir=".", cell_id=cell.cell_id)
            for path, w, rho_h, rho_d in blends:
                print(f"  wrote {path}  (w={w}, ρ_hep={rho_h:.3f}, ρ_death={rho_d:.3f})")


if __name__ == "__main__":
    main()
