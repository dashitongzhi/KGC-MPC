"""Perturbation prediction, target prioritization, and unconstrained control baselines for Paper 3 (KGC-MPC)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kgc_mpc import get_device


def evaluate_perturbation_predictions(
    y_true_deltas: np.ndarray,
    y_pred_deltas: np.ndarray,
    de_indices_list: List[np.ndarray],
) -> Dict[str, float]:
    """Evaluate perturbation response prediction across all genes and Top-20 DE genes."""
    all_mse = []
    all_r = []
    de_mse = []
    de_r = []
    de_dir = []
    for i in range(len(y_true_deltas)):
        yt = y_true_deltas[i]
        yp = y_pred_deltas[i]
        all_mse.append(float(np.mean((yt - yp) ** 2)))
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            all_r.append(float(np.corrcoef(yt, yp)[0, 1]))
        else:
            all_r.append(0.0)

        idx = de_indices_list[i]
        yt_d = yt[idx]
        yp_d = yp[idx]
        de_mse.append(float(np.mean((yt_d - yp_d) ** 2)))
        if np.std(yt_d) > 1e-8 and np.std(yp_d) > 1e-8:
            de_r.append(float(np.corrcoef(yt_d, yp_d)[0, 1]))
        else:
            de_r.append(0.0)
        de_dir.append(float(np.mean(np.sign(yt_d) == np.sign(yp_d))))

    return {
        "all_mse": float(np.mean(all_mse)),
        "all_pearson": float(np.mean(all_r)),
        "de20_mse": float(np.mean(de_mse)),
        "de20_pearson": float(np.mean(de_r)),
        "de20_dir_acc": float(np.mean(de_dir)),
    }


def compute_linear_additive_baseline(
    ctrl_mean: np.ndarray,
    train_deltas_by_unit: Dict[str, np.ndarray],
    test_pert_combos: List[Tuple[str, ...]],
    test_doses: Optional[np.ndarray] = None,
    dose_unit_slopes: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> np.ndarray:
    """Linear-Additive baseline: sum single-perturbation shifts or log-linear dose extrapolation."""
    n_genes = len(ctrl_mean)
    preds = np.zeros((len(test_pert_combos), n_genes), dtype=np.float32)
    for i, units in enumerate(test_pert_combos):
        if dose_unit_slopes is not None and test_doses is not None and len(units) == 1:
            u_name = units[0]
            if u_name in dose_unit_slopes:
                slope, intercept = dose_unit_slopes[u_name]
                d_val = np.log10(1.0 + float(test_doses[i]) / 10.0)
                preds[i] = slope * d_val + intercept
                continue
        delta = np.zeros(n_genes, dtype=np.float32)
        for u in units:
            if u in train_deltas_by_unit:
                delta += train_deltas_by_unit[u]
        preds[i] = delta
    return preds


class UnconstrainedEDMD:
    """Unconstrained Extended Dynamic Mode Decomposition with Control (DMDc) in PCA space."""

    def __init__(self, latent_dim: int = 64, reg: float = 1e-3, horizon: int = 3) -> None:
        self.latent_dim = latent_dim
        self.reg = reg
        self.horizon = horizon
        self.W_pca: Optional[np.ndarray] = None
        self.ctrl_mean: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None

    def fit(self, x_ctrl: np.ndarray, x_pert: np.ndarray, u_pert: np.ndarray) -> "UnconstrainedEDMD":
        self.ctrl_mean = x_ctrl.mean(axis=0)
        n_sub = min(len(x_pert), 12000)
        pooled = np.vstack([x_ctrl[: min(len(x_ctrl), 4000)] - self.ctrl_mean, x_pert[:n_sub] - self.ctrl_mean])
        _, _, vt = np.linalg.svd(pooled, full_matrices=False)
        r = min(self.latent_dim, vt.shape[0])
        self.W_pca = vt[:r].T  # (G, r)

        z0 = (x_ctrl[: min(len(x_ctrl), len(x_pert))] - self.ctrl_mean) @ self.W_pca
        rng = np.random.default_rng(42)
        c_idx = rng.integers(0, len(z0), size=len(x_pert))
        Z_prev = z0[c_idx]
        Z_next = (x_pert - self.ctrl_mean) @ self.W_pca
        # Unconstrained pseudoinverse regression on [Z_prev, u_pert] -> Z_next
        Omega = np.hstack([Z_prev, u_pert])
        G = Omega.T @ Omega + self.reg * np.eye(Omega.shape[1])
        AB = np.linalg.solve(G, Omega.T @ Z_next).T  # (r, r + M)
        self.A = AB[:, :r]
        self.B = AB[:, r:]
        return self

    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def predict_delta(self, u_test: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        steps = self.horizon if steps is None else steps
        z = np.zeros((len(u_test), self.A.shape[0]), dtype=np.float64)
        for _ in range(steps):
            z = z @ self.A.T + u_test @ self.B.T
        return (z @ self.W_pca.T).astype(np.float32)


class CPALatentAE(nn.Module):
    """Compositional Perturbation Autoencoder (CPA-style) baseline."""

    def __init__(self, n_genes: int, n_perts: int, latent_dim: int = 64, dose_mode: bool = False) -> None:
        super().__init__()
        self.dose_mode = dose_mode
        self.enc = nn.Sequential(nn.Linear(n_genes, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.pert_emb = nn.Linear(n_perts, latent_dim, bias=False)
        self.dosers = nn.Sequential(nn.Linear(n_perts, 64), nn.ReLU(), nn.Linear(64, n_perts), nn.Sigmoid())
        self.dec = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, n_genes))

    def forward(self, x_ctrl: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        z = self.enc(x_ctrl)
        if self.dose_mode:
            u_eff = u * self.dosers(u)
        else:
            u_eff = u
        z_p = z + self.pert_emb(u_eff)
        return self.dec(z_p)


def fit_and_predict_cpa(
    x_ctrl: np.ndarray,
    x_pert: np.ndarray,
    u_pert: np.ndarray,
    u_test: np.ndarray,
    dose_mode: bool = False,
    epochs: int = 40,
    batch_size: int = 512,
    seed: int = 42,
    device: Optional[str] = None,
) -> np.ndarray:
    torch.manual_seed(seed)
    dev = get_device(device)
    ctrl_mean = x_ctrl.mean(axis=0).astype(np.float32)
    xc_t = torch.tensor(x_ctrl - ctrl_mean, dtype=torch.float32, device=dev)
    xp_t = torch.tensor(x_pert - ctrl_mean, dtype=torch.float32, device=dev)
    up_t = torch.tensor(u_pert, dtype=torch.float32, device=dev)

    model = CPALatentAE(x_ctrl.shape[1], u_pert.shape[1], latent_dim=64, dose_mode=dose_mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xp_t), device=dev)
        for s in range(0, len(xp_t), batch_size):
            bi = perm[s : s + batch_size]
            ci = torch.randint(0, len(xc_t), (len(bi),), device=dev)
            pred = model(xc_t[ci], up_t[bi])
            rec = model(xc_t[ci], torch.zeros_like(up_t[bi]))
            loss = F.mse_loss(pred, xp_t[bi]) + 0.3 * F.mse_loss(rec, xc_t[ci])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        x0_m = torch.zeros((len(u_test), x_ctrl.shape[1]), dtype=torch.float32, device=dev)
        ut_t = torch.tensor(u_test, dtype=torch.float32, device=dev)
        return model(x0_m, ut_t).cpu().numpy()


class GEARSCoExprGNN(nn.Module):
    """Co-expression graph perturbation network (GEARS-style compositional message passing)."""

    def __init__(self, n_genes: int, n_perts: int, pert_feats: np.ndarray, hidden_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer("pf", torch.tensor(pert_feats, dtype=torch.float32))
        fdim = pert_feats.shape[1]
        self.pert_enc = nn.Sequential(nn.Linear(fdim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cross_gene = nn.Sequential(nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Linear(128, hidden_dim))
        self.dec = nn.Sequential(nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, n_genes))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # Weighted sum over perturbed units' embeddings + cross-gene nonlinear interaction
        unit_emb = self.pert_enc(self.pf)  # (M, h)
        agg = u @ unit_emb  # (B, h)
        h = agg + self.cross_gene(agg)
        return self.dec(h)


def fit_and_predict_gears_gnn(
    x_ctrl: np.ndarray,
    x_pert: np.ndarray,
    u_pert: np.ndarray,
    u_test: np.ndarray,
    pert_feats: np.ndarray,
    epochs: int = 60,
    seed: int = 42,
    device: Optional[str] = None,
) -> np.ndarray:
    torch.manual_seed(seed)
    dev = get_device(device)
    ctrl_mean = x_ctrl.mean(axis=0).astype(np.float32)

    cond_ids, cond_inv = np.unique(u_pert, axis=0, return_inverse=True)
    cond_deltas = np.zeros((len(cond_ids), x_ctrl.shape[1]), dtype=np.float32)
    for ci in range(len(cond_ids)):
        cond_deltas[ci] = x_pert[cond_inv == ci].mean(axis=0) - ctrl_mean

    cu_t = torch.tensor(cond_ids, dtype=torch.float32, device=dev)
    cd_t = torch.tensor(cond_deltas, dtype=torch.float32, device=dev)
    model = GEARSCoExprGNN(x_ctrl.shape[1], u_pert.shape[1], pert_feats).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(cu_t), device=dev)
        for s in range(0, len(cu_t), 64):
            bi = perm[s : s + 64]
            pred = model(cu_t[bi])
            loss = F.mse_loss(pred, cd_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        ut_t = torch.tensor(u_test, dtype=torch.float32, device=dev)
        return model(ut_t).cpu().numpy()
