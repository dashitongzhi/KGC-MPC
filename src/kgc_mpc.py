"""Schur-Stable Bilinear Koopman System, Controllability Gramian Target Prioritization, and Safety-Constrained MPC (KGC-MPC)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def polar_orthogonal(mat: torch.Tensor) -> torch.Tensor:
    """Differentiable nearest orthogonal matrix on Stiefel manifold via SVD polar retraction."""
    u, _, vh = torch.linalg.svd(mat, full_matrices=False)
    return u @ vh


class KGCSystem(nn.Module):
    """Schur-stable bilinear Koopman dynamical system for perturbation and dose-response control."""

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        latent_dim: int = 64,
        bilinear_rank: int = 12,
        horizon: int = 3,
        gamma_schur: float = 0.95,
        dose_mode: bool = False,
        bilin_max_gain: Optional[float] = None,
        stiefel_contraction: bool = True,
        pca_basis: Optional[np.ndarray] = None,
        ctrl_mean: Optional[np.ndarray] = None,
        gene_pert_features: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_perts = int(n_perts)
        self.latent_dim = int(latent_dim)
        self.bilinear_rank = int(bilinear_rank)
        self.horizon = int(horizon)
        self.gamma_schur = float(gamma_schur)
        self.dose_mode = bool(dose_mode)
        self.bilin_max_gain = None if bilin_max_gain is None else float(bilin_max_gain)
        self.stiefel_contraction = bool(stiefel_contraction)

        # Control-mean centering buffer
        if ctrl_mean is not None:
            self.register_buffer("ctrl_mean", torch.tensor(ctrl_mean, dtype=torch.float32))
        else:
            self.register_buffer("ctrl_mean", torch.zeros(n_genes, dtype=torch.float32))

        # Orthogonal PCA anchor + residual nonlinear Koopman lifting observable psi(x)
        if pca_basis is not None:
            w_init = torch.tensor(pca_basis[:, :latent_dim], dtype=torch.float32)  # (G, r)
        else:
            w_init = torch.randn(n_genes, latent_dim, dtype=torch.float32) * (1.0 / np.sqrt(n_genes))
            w_init, _ = torch.linalg.qr(w_init)
        self.W_lin = nn.Parameter(w_init.clone())
        self.C_dec = nn.Parameter(w_init.clone())  # (G, r) decoder: x_hat = z @ C^T + ctrl_mean

        self.obs_mlp = nn.Sequential(
            nn.Linear(n_genes, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )
        nn.init.zeros_(self.obs_mlp[-1].weight)
        nn.init.zeros_(self.obs_mlp[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        # Schur-stable A = U diag(gamma * tanh(sigma)) V^T parameterization (or unconstrained ablation)
        if self.stiefel_contraction:
            self.U_raw = nn.Parameter(torch.eye(latent_dim, dtype=torch.float32) + 0.02 * torch.randn(latent_dim, latent_dim))
            self.V_raw = nn.Parameter(torch.eye(latent_dim, dtype=torch.float32) + 0.02 * torch.randn(latent_dim, latent_dim))
            self.sigma_raw = nn.Parameter(torch.full((latent_dim,), 0.85, dtype=torch.float32))
            self.A_unconstrained = None
        else:
            self.U_raw = None
            self.V_raw = None
            self.sigma_raw = None
            self.A_unconstrained = nn.Parameter(
                0.85 * torch.eye(latent_dim, dtype=torch.float32) + 0.02 * torch.randn(latent_dim, latent_dim)
            )

        # Actuation matrix B: co-expression aligned + per-perturbation residual
        if gene_pert_features is not None:
            self.register_buffer("pert_feat", torch.tensor(gene_pert_features, dtype=torch.float32))  # (M, f)
            feat_dim = int(gene_pert_features.shape[1])
            self.W_act = nn.Linear(feat_dim, latent_dim, bias=False)
            nn.init.normal_(self.W_act.weight, std=0.05)
        else:
            self.pert_feat = None
            self.W_act = None
        self.B_delta = nn.Parameter(torch.randn(latent_dim, n_perts, dtype=torch.float32) * 0.02)

        # Dose saturation (Hill-log) parameters alpha_j >= 0
        self.alpha_raw = nn.Parameter(torch.full((n_perts,), -1.5, dtype=torch.float32))

        # Low-rank bilinear epistasis / drug-state coupling: sum_l (g_l^T h(u)) (p_l^T z) r_l
        self.bilin_G = nn.Parameter(torch.randn(bilinear_rank, n_perts, dtype=torch.float32) * 0.03)
        self.bilin_P = nn.Parameter(torch.randn(bilinear_rank, latent_dim, dtype=torch.float32) * 0.05)
        self.bilin_R = nn.Parameter(torch.randn(bilinear_rank, latent_dim, dtype=torch.float32) * 0.05)

    def get_A(self) -> torch.Tensor:
        """Construct strictly Schur-stable transition matrix A with spectral norm <= gamma_schur < 1."""
        if not self.stiefel_contraction and self.A_unconstrained is not None:
            return self.A_unconstrained
        u = polar_orthogonal(self.U_raw)
        v = polar_orthogonal(self.V_raw)
        s = self.gamma_schur * torch.tanh(self.sigma_raw)
        return (u * s.unsqueeze(0)) @ v.t()

    def get_B(self) -> torch.Tensor:
        """Construct actuation matrix B in R^{r x M}."""
        if self.pert_feat is not None and self.W_act is not None:
            b_prior = self.W_act(self.pert_feat).t()  # (r, M)
            return b_prior + self.B_delta
        return self.B_delta

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - self.ctrl_mean
        zero_in = torch.zeros((1, self.n_genes), dtype=xc.dtype, device=xc.device)
        res_corr = self.obs_mlp(xc) - self.obs_mlp(zero_in)
        return xc @ self.W_lin + self.res_scale * res_corr

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.C_dec.t() + self.ctrl_mean

    def saturate_input(self, u: torch.Tensor) -> torch.Tensor:
        """Apply Hill-log saturation for dose inputs or pass through binary perturbation indicators."""
        if not self.dose_mode:
            return u
        alpha = F.softplus(self.alpha_raw).unsqueeze(0)
        # u is expected as log10(1 + dose_nM / 10.0) when dose_mode=True
        return u / (1.0 + alpha * u)

    def step(
        self,
        z: torch.Tensor,
        h_u: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        B: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if A is None:
            A = self.get_A()
        if B is None:
            B = self.get_B()
        z_lin = z @ A.t() + h_u @ B.t()
        if self.bilinear_rank <= 0:
            return z_lin
        # Bilinear interaction: (batch, q) * (batch, q) -> (batch, r)
        g_u = h_u @ self.bilin_G.t()
        p_z = z @ self.bilin_P.t()
        bilin = (g_u * p_z) @ self.bilin_R
        if self.bilin_max_gain is not None:
            z_norm = torch.linalg.norm(z, dim=-1, keepdim=True) + 1e-6
            b_norm = torch.linalg.norm(bilin, dim=-1, keepdim=True) + 1e-8
            bilin = bilin * torch.clamp((self.bilin_max_gain * z_norm) / b_norm, max=1.0)
        return z_lin + bilin

    def rollout(
        self,
        z0: torch.Tensor,
        u: torch.Tensor,
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        steps = self.horizon if steps is None else int(steps)
        A = self.get_A()
        B = self.get_B()
        h_u = self.saturate_input(u)
        traj = [z0]
        zt = z0
        for _ in range(steps):
            zt = self.step(zt, h_u, A=A, B=B)
            traj.append(zt)
        return zt, traj

    def forward_x(
        self,
        x0: torch.Tensor,
        u: torch.Tensor,
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        z0 = self.encode(x0)
        zT, traj = self.rollout(z0, u, steps=steps)
        x_hat = self.decode(zT)
        return x_hat, zT, traj

    @torch.no_grad()
    def spectral_radius(self) -> float:
        A_np = self.get_A().detach().cpu().numpy()
        return float(np.max(np.abs(np.linalg.eigvals(A_np))))

    @torch.no_grad()
    def extract_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A_np = self.get_A().detach().cpu().numpy().astype(np.float64)
        B_np = self.get_B().detach().cpu().numpy().astype(np.float64)
        C_np = self.C_dec.detach().cpu().numpy().astype(np.float64)
        alpha_np = F.softplus(self.alpha_raw).detach().cpu().numpy().astype(np.float64)
        return A_np, B_np, C_np, alpha_np


def fit_kgc_system(
    x_ctrl: np.ndarray,
    x_pert: np.ndarray,
    u_pert: np.ndarray,
    pert_names: List[str],
    dose_mode: bool = False,
    latent_dim: int = 64,
    bilinear_rank: int = 12,
    horizon: int = 3,
    gamma_schur: float = 0.95,
    bilin_max_gain: Optional[float] = None,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 1.5e-3,
    seed: int = 42,
    stiefel_contraction: bool = True,
    gene_pert_features: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> KGCSystem:
    """Train KGCSystem on paired control-to-perturbed single-cell transitions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = get_device(device)

    n_genes = x_ctrl.shape[1]
    n_perts = u_pert.shape[1]
    ctrl_mean = x_ctrl.mean(axis=0).astype(np.float32)

    # Compute PCA basis from pooled centered training data (fast randomized SVD)
    n_sub = min(len(x_pert), 15000)
    idx_sub = np.random.choice(len(x_pert), size=n_sub, replace=False) if len(x_pert) > n_sub else np.arange(len(x_pert))
    pooled = np.vstack([x_ctrl[: min(len(x_ctrl), 5000)] - ctrl_mean, x_pert[idx_sub] - ctrl_mean])
    U_s, S_s, Vt_s = np.linalg.svd(pooled, full_matrices=False)
    r_eff = min(latent_dim, Vt_s.shape[0])
    pca_basis = Vt_s[:r_eff].T.astype(np.float32)
    if r_eff < latent_dim:
        pad = np.zeros((n_genes, latent_dim - r_eff), dtype=np.float32)
        pca_basis = np.hstack([pca_basis, pad])

    model = KGCSystem(
        n_genes=n_genes,
        n_perts=n_perts,
        latent_dim=latent_dim,
        bilinear_rank=bilinear_rank,
        horizon=horizon,
        gamma_schur=gamma_schur,
        dose_mode=dose_mode,
        bilin_max_gain=bilin_max_gain,
        stiefel_contraction=stiefel_contraction,
        pca_basis=pca_basis,
        ctrl_mean=ctrl_mean,
        gene_pert_features=gene_pert_features,
    ).to(dev)

    # Also build perturbation-condition pseudo-bulk targets for distribution matching
    cond_ids, cond_inv = np.unique(u_pert, axis=0, return_inverse=True)
    cond_means = np.zeros((len(cond_ids), n_genes), dtype=np.float32)
    for ci in range(len(cond_ids)):
        cond_means[ci] = x_pert[cond_inv == ci].mean(axis=0)

    x_ctrl_t = torch.tensor(x_ctrl, dtype=torch.float32, device=dev)
    x_pert_t = torch.tensor(x_pert, dtype=torch.float32, device=dev)
    u_pert_t = torch.tensor(u_pert, dtype=torch.float32, device=dev)
    cond_u_t = torch.tensor(cond_ids, dtype=torch.float32, device=dev)
    cond_m_t = torch.tensor(cond_means, dtype=torch.float32, device=dev)
    cond_inv_t = torch.tensor(cond_inv, dtype=torch.long, device=dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    n_samples = len(x_pert)
    n_ctrl = len(x_ctrl)

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_samples, device=dev)
        for start in range(0, n_samples, batch_size):
            b_idx = perm[start : start + batch_size]
            c_idx = torch.randint(0, n_ctrl, (len(b_idx),), device=dev)
            x0_b = x_ctrl_t[c_idx]
            xt_b = x_pert_t[b_idx]
            u_b = u_pert_t[b_idx]
            target_mean_b = cond_m_t[cond_inv_t[b_idx]]

            # Autoencoder reconstruction consistency on both ctrl & pert cells
            z_ctrl = model.encode(x0_b)
            z_pert = model.encode(xt_b)
            rec_loss = F.mse_loss(model.decode(z_ctrl), x0_b) + F.mse_loss(model.decode(z_pert), xt_b)

            # Koopman forward rollout from z_ctrl -> z_T matching target perturbed phenotype
            zT, traj = model.rollout(z_ctrl, u_b, steps=horizon)
            x_pred = model.decode(zT)
            pred_loss = F.mse_loss(x_pred, target_mean_b)

            # Unforced stability consistency: control cell stays near control under u=0
            z_zero_T, _ = model.rollout(z_ctrl, torch.zeros_like(u_b), steps=horizon)
            zero_loss = F.mse_loss(z_zero_T, z_ctrl)

            # Latent alignment between forward trajectory endpoint and encoded target
            latent_loss = F.mse_loss(zT, z_pert.detach())

            # Condition-level batch distribution alignment
            loss = (
                pred_loss
                + 0.35 * rec_loss
                + 0.25 * latent_loss
                + 0.15 * zero_loss
                + 5e-3 * F.mse_loss(model.C_dec, model.W_lin.detach())
                + 1e-3 * (model.B_delta.pow(2).mean() + model.bilin_G.pow(2).mean())
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        # Also do a condition-level clean pass for sharp pseudo-bulk trajectory matching
        c_perm = torch.randperm(len(cond_ids), device=dev)
        for start in range(0, len(cond_ids), min(128, max(16, len(cond_ids)))):
            ci = c_perm[start : start + 128]
            z0_mean = model.encode(x_ctrl_t[: min(n_ctrl, 2048)].mean(dim=0, keepdim=True))
            z0_rep = z0_mean.expand(len(ci), -1)
            zT_c, _ = model.rollout(z0_rep, cond_u_t[ci], steps=horizon)
            x_hat_c = model.decode(zT_c)
            loss_c = F.mse_loss(x_hat_c, cond_m_t[ci]) + 0.05 * F.smooth_l1_loss(x_hat_c, cond_m_t[ci])
            opt.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        scheduler.step()

    model.eval()
    return model


def compute_nonnormal_diagnostics(A: np.ndarray, max_step: int = 20) -> Dict[str, float]:
    """Compute spectral radius, L2 norm, unit-eigenvector condition number, power norms, and Kreiss bound."""
    A64 = np.asarray(A, dtype=np.float64)
    r = A64.shape[0]
    eigvals, V = np.linalg.eig(A64)
    rho = float(np.max(np.abs(eigvals)))
    sigma_max = float(np.linalg.svd(A64, compute_uv=False)[0])

    col_norms = np.linalg.norm(V, axis=0, keepdims=True)
    V_unit = V / np.maximum(col_norms, 1e-12)
    s_v = np.linalg.svd(V_unit, compute_uv=False)
    eigvec_cond = float(s_v[0] / s_v[-1]) if s_v[-1] > 1e-12 else 1e12

    A_pow = np.eye(r, dtype=np.float64)
    max_power_norm = 0.0
    for _ in range(max_step):
        A_pow = A64 @ A_pow
        max_power_norm = max(max_power_norm, float(np.linalg.svd(A_pow, compute_uv=False)[0]))

    thetas = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radii = (1.005, 1.02, 1.05, 1.1, 1.2, 1.5)
    I_r = np.eye(r, dtype=np.complex128)
    kreiss_lb = 1.0
    for rad in radii:
        for th in thetas:
            z = rad * np.exp(1j * th)
            try:
                s_min = float(np.linalg.svd(z * I_r - A64, compute_uv=False)[-1])
                val = (rad - 1.0) / max(s_min, 1e-12)
                if val > kreiss_lb:
                    kreiss_lb = val
            except np.linalg.LinAlgError:
                continue

    return {
        "rho": rho,
        "sigma_max": sigma_max,
        "eigvec_cond": eigvec_cond,
        "max_power_norm": max_power_norm,
        "kreiss_lb": float(kreiss_lb),
    }


def compute_controllability_gramian(
    A: np.ndarray,
    B: np.ndarray,
    horizon: int = 6,
    reg_R: float = 1e-2,
) -> np.ndarray:
    """Exact finite-horizon controllability Gramian W_c(H) = sum_{t=0}^{H-1} A^t B B^T (A^T)^t."""
    r = A.shape[0]
    W = np.zeros((r, r), dtype=np.float64)
    A_pow = np.eye(r, dtype=np.float64)
    BBt = B @ B.T
    for _ in range(horizon):
        W += A_pow @ BBt @ A_pow.T
        A_pow = A @ A_pow
    W = 0.5 * (W + W.T) + reg_R * np.eye(r, dtype=np.float64)
    return W


def score_targets_gramian(
    A: np.ndarray,
    B: np.ndarray,
    z_src: np.ndarray,
    z_tgt: np.ndarray,
    horizon: int = 6,
    lam_r: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute directional minimum control energy E_min(j) and prioritization score for each actuator b_j."""
    r, M = B.shape
    # Accumulated reachability operator C_H = sum_{t=0}^{H-1} A^t
    C_H = np.zeros((r, r), dtype=np.float64)
    A_pow = np.eye(r, dtype=np.float64)
    for _ in range(horizon):
        C_H += A_pow
        A_pow = A @ A_pow

    delta_z = (z_tgt - A_pow @ z_src).astype(np.float64)
    dz_norm2 = float(np.dot(delta_z, delta_z)) + 1e-12
    reach_B = C_H @ B  # (r, M)
    align = (delta_z @ reach_B)  # signed projection onto steering direction
    # Effective directional energy E_min(j) = ||dz||^4 / (max(0, dz^T C_H b_j)^2 + eps) + lam * ||b_j||^2
    pos_align = np.maximum(0.0, align)
    b_norms2 = np.sum(B ** 2, axis=0)
    energy = (dz_norm2 ** 2) / (pos_align ** 2 + 1e-6) + lam_r * b_norms2
    # Higher score = better therapeutic reprogramming priority (higher positive alignment / lower energy)
    score = align / np.sqrt(b_norms2 + lam_r)
    return score, energy


def select_cocktail_submodular(
    A: np.ndarray,
    B: np.ndarray,
    budget_k: int,
    horizon: int = 6,
    eps: float = 1e-3,
    candidate_mask: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[float]]:
    """Submodular greedy selection maximizing log det(W_c(S) + eps I) with (1-1/e) guarantee."""
    r, M = B.shape
    C_H_list = []
    A_pow = np.eye(r, dtype=np.float64)
    for _ in range(horizon):
        C_H_list.append(A_pow.copy())
        A_pow = A @ A_pow

    # Precompute Gramian rank-H atom G_j = sum_t (A^t b_j)(A^t b_j)^T
    atoms = []
    for j in range(M):
        bj = B[:, j : j + 1]
        gj = np.zeros((r, r), dtype=np.float64)
        for Ap in C_H_list:
            v = Ap @ bj
            gj += v @ v.T
        atoms.append(0.5 * (gj + gj.T))

    candidates = np.where(candidate_mask)[0].tolist() if candidate_mask is not None else list(range(M))
    W_cur = eps * np.eye(r, dtype=np.float64)
    _, cur_logdet = np.linalg.slogdet(W_cur)
    selected: List[int] = []
    gains: List[float] = []

    for _ in range(min(budget_k, len(candidates))):
        best_j = -1
        best_val = -1e18
        W_inv = np.linalg.inv(W_cur)
        for j in candidates:
            if j in selected:
                continue
            # Exact logdet rank-update / trace-bounded exact check
            _, val = np.linalg.slogdet(W_cur + atoms[j])
            if val > best_val:
                best_val = val
                best_j = j
        if best_j < 0:
            break
        selected.append(best_j)
        gains.append(float(best_val - cur_logdet))
        W_cur = W_cur + atoms[best_j]
        cur_logdet = best_val
    return selected, gains


def solve_safety_mpc(
    A: np.ndarray,
    B: np.ndarray,
    z0: np.ndarray,
    z_tgt: np.ndarray,
    P_offtarget: np.ndarray,
    delta_safe: float = 0.8,
    u_max: float = 1.0,
    horizon: int = 8,
    q_weight: float = 1.0,
    r_weight: float = 0.05,
    n_iters: int = 350,
    lr: float = 0.08,
    penalty_rho: float = 450.0,
    enforce_constraints: bool = True,
    z_tube_ref: Optional[np.ndarray] = None,
    solver_mode: str = "fista_al",
) -> Dict[str, object]:
    """Solve finite-horizon MPC with DARE terminal cost, input box bounds, and off-target safety tube."""
    z_ref = z0 if z_tube_ref is None else np.asarray(z_tube_ref, dtype=np.float64)
    if enforce_constraints and solver_mode == "fista_al":
        solver = CondensedSafeMPCSolver(
            A,
            B,
            P_offtarget,
            horizon=horizon,
            q_weight=q_weight,
            r_weight=r_weight,
            backend="numpy",
        )
        return solver.solve(
            z0=z0,
            z_tgt=z_tgt,
            delta_safe=delta_safe,
            u_max=u_max,
            z_tube_ref=z_ref,
            enforce_feas_projection=True,
        )
    if enforce_constraints and solver_mode != "adam_penalty":
        raise ValueError(f"Unsupported solver_mode: {solver_mode}")
    r, M = B.shape
    Q = q_weight * np.eye(r, dtype=np.float64)
    R = r_weight * np.eye(M, dtype=np.float64)
    try:
        P_term = solve_discrete_are(A, B, Q, R)
    except Exception:
        P_term = 5.0 * Q

    if not enforce_constraints:
        # Closed-form unconstrained finite-horizon LQR backward Riccati recursion
        P_list = [None] * (horizon + 1)
        K_list = [None] * horizon
        P_list[horizon] = P_term
        for t in range(horizon - 1, -1, -1):
            Pt1 = P_list[t + 1]
            K = np.linalg.solve(R + B.T @ Pt1 @ B, B.T @ Pt1 @ A)
            K_list[t] = K
            P_list[t] = Q + A.T @ Pt1 @ (A - B @ K)
        # Forward rollout around shifted equilibrium z_tgt
        z_traj = [z0.copy()]
        u_traj = []
        zt = z0.copy()
        u_ss = np.linalg.lstsq(B, (np.eye(r) - A) @ z_tgt, rcond=None)[0]
        for t in range(horizon):
            ut = u_ss - K_list[t] @ (zt - z_tgt)
            u_traj.append(ut)
            zt = A @ zt + B @ ut
            z_traj.append(zt.copy())
        z_arr = np.stack(z_traj, axis=0)
        u_arr = np.stack(u_traj, axis=0)
    else:
        # Torch accelerated projected gradient + smooth safety-tube penalty for box + tube constraints
        with torch.enable_grad():
            A_t = torch.tensor(A, dtype=torch.float64)
            B_t = torch.tensor(B, dtype=torch.float64)
            z0_t = torch.tensor(z0, dtype=torch.float64)
            zref_t = torch.tensor(z_ref, dtype=torch.float64)
            ztgt_t = torch.tensor(z_tgt, dtype=torch.float64)
            P_t = torch.tensor(P_term, dtype=torch.float64)
            Po_t = torch.tensor(P_offtarget, dtype=torch.float64)

            u_var = nn.Parameter(torch.full((horizon, M), 0.10 * u_max, dtype=torch.float64))
            opt = torch.optim.Adam([u_var], lr=lr)
            for _ in range(n_iters):
                opt.zero_grad()
                zt = z0_t
                cost = torch.tensor(0.0, dtype=torch.float64)
                for t in range(horizon):
                    ut = u_var[t]
                    err = zt - ztgt_t
                    cost = cost + q_weight * torch.sum(err * err) + r_weight * torch.sum(ut * ut)
                    zt = A_t @ zt + B_t @ ut
                    off = Po_t @ (zt - zref_t)
                    off_norm = torch.sqrt(torch.sum(off * off) + 1e-12)
                    viol = torch.relu(off_norm - delta_safe)
                    cost = cost + penalty_rho * (viol * viol)
                err_T = zt - ztgt_t
                cost = cost + err_T @ P_t @ err_T
                cost.backward()
                opt.step()
                with torch.no_grad():
                    u_var.clamp_(0.0, u_max)

        with torch.no_grad():
            u_arr = u_var.detach().numpy()
            z_traj = [z0.copy()]
            zt = z0.copy()
            for t in range(horizon):
                zt = A @ zt + B @ u_arr[t]
                z_traj.append(zt.copy())
            z_arr = np.stack(z_traj, axis=0)

    off_norms = np.array([np.linalg.norm(P_offtarget @ (z_arr[t] - z_ref)) for t in range(1, horizon + 1)])
    target_err = float(np.linalg.norm(z_arr[-1] - z_tgt) / (np.linalg.norm(z0 - z_tgt) + 1e-8))
    max_off = float(np.max(off_norms))
    viol_rate = float(np.mean(off_norms > delta_safe))
    total_energy = float(np.sum(u_arr ** 2))
    return {
        "u_traj": u_arr,
        "z_traj": z_arr,
        "off_norms": off_norms,
        "max_offtarget": max_off,
        "violation_rate": viol_rate,
        "relative_target_error": target_err,
        "control_energy": total_energy,
    }


class CondensedSafeMPCSolver:
    """Real-time condensed FISTA Augmented-Lagrangian MPC solver with DARE terminal cost and safety tube."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        P_offtarget: np.ndarray,
        horizon: int = 8,
        q_weight: float = 1.0,
        r_weight: float = 0.05,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        backend: str = "torch",
    ) -> None:
        self.horizon = int(horizon)
        self.r_dim, self.k_dim = B.shape
        self.q_weight = float(q_weight)
        self.r_weight = float(r_weight)
        self.backend = str(backend).lower()
        if self.backend not in ("torch", "numpy"):
            raise ValueError(f"Unsupported backend: {backend}")
        self.device = torch.device("cpu") if self.backend == "numpy" else get_device(device)
        self.dtype = dtype

        A64 = np.asarray(A, dtype=np.float64)
        B64 = np.asarray(B, dtype=np.float64)
        Po64 = np.asarray(P_offtarget, dtype=np.float64)
        Q64 = self.q_weight * np.eye(self.r_dim, dtype=np.float64)
        R64 = self.r_weight * np.eye(self.k_dim, dtype=np.float64)
        try:
            P_term = solve_discrete_are(A64, B64, Q64, R64)
        except Exception:
            P_term = 5.0 * Q64

        H = self.horizon
        r, k = self.r_dim, self.k_dim
        Phi = np.zeros((H * r, r), dtype=np.float64)
        Gamma = np.zeros((H * r, H * k), dtype=np.float64)
        A_pows = [np.eye(r, dtype=np.float64)]
        for _ in range(H):
            A_pows.append(A64 @ A_pows[-1])
        for t in range(H):
            Phi[t * r : (t + 1) * r, :] = A_pows[t + 1]
            for j in range(t + 1):
                Gamma[t * r : (t + 1) * r, j * k : (j + 1) * k] = A_pows[t - j] @ B64

        Q_bar_diag = [Q64] * (H - 1) + [P_term]
        Q_bar = np.zeros((H * r, H * r), dtype=np.float64)
        for t, Qt in enumerate(Q_bar_diag):
            Q_bar[t * r : (t + 1) * r, t * r : (t + 1) * r] = Qt
        R_bar = np.kron(np.eye(H, dtype=np.float64), R64)

        H_qp = Gamma.T @ Q_bar @ Gamma + R_bar
        H_qp = 0.5 * (H_qp + H_qp.T)
        L_qp = float(np.linalg.eigvalsh(H_qp)[-1])

        P_blk = np.kron(np.eye(H, dtype=np.float64), Po64)
        Phi_perp = P_blk @ Phi
        Gamma_perp = P_blk @ Gamma
        gamma_perp_norm2 = float(np.linalg.norm(Gamma_perp, ord=2) ** 2)

        self.A_np = A64
        self.B_np = B64
        self.P_off_np = Po64
        self.P_term_np = P_term
        self.H_qp_np64 = H_qp
        self.Phi_np64 = Phi
        self.Gamma_np64 = Gamma
        self.Q_bar_np64 = Q_bar
        self.Q_bar_Gamma_np64 = Q_bar @ Gamma
        self.Phi_perp_np64 = Phi_perp
        self.Gamma_perp_np64 = Gamma_perp
        self.L_qp = L_qp
        self.gamma_perp_norm2 = gamma_perp_norm2

        # Float32 BLAS buffers for fast single-trajectory CPU execution
        self.Po32 = Po64.astype(np.float32)
        self.Phi32 = Phi.astype(np.float32)
        self.Gamma32 = Gamma.astype(np.float32)
        self.Q_bar32 = Q_bar.astype(np.float32)
        self.Q_bar_Gamma_T32 = (Q_bar @ Gamma).T.astype(np.float32)
        self.two_H_qp32 = (2.0 * H_qp).astype(np.float32)
        self.Phi_perp32 = Phi_perp.astype(np.float32)
        self.Gamma_perp32 = Gamma_perp.astype(np.float32)
        self.Gamma_perp_T32 = Gamma_perp.T.astype(np.float32)

        if self.backend == "torch":
            self.A_t = torch.tensor(A64, dtype=dtype, device=self.device)
            self.B_t = torch.tensor(B64, dtype=dtype, device=self.device)
            self.P_off_t = torch.tensor(Po64, dtype=dtype, device=self.device)
            self.Phi_t = torch.tensor(Phi, dtype=dtype, device=self.device)
            self.Gamma_t = torch.tensor(Gamma, dtype=dtype, device=self.device)
            self.Q_bar_Gamma_t = torch.tensor(Q_bar @ Gamma, dtype=dtype, device=self.device)
            self.H_qp_t = torch.tensor(H_qp, dtype=dtype, device=self.device)
            self.Phi_perp_t = torch.tensor(Phi_perp, dtype=dtype, device=self.device)
            self.Gamma_perp_t = torch.tensor(Gamma_perp, dtype=dtype, device=self.device)
            self.Q_bar_t = torch.tensor(Q_bar, dtype=dtype, device=self.device)

    def _solve_numpy(
        self,
        z0: np.ndarray,
        z_tgt: np.ndarray,
        delta_safe: float,
        u_max: float,
        z_tube_ref: Optional[np.ndarray],
        u_init: Optional[np.ndarray],
        outer_iters: int,
        inner_iters: int,
        rho_init: float,
        rho_mult: float,
        enforce_feas_projection: bool,
        tol: float,
    ) -> Dict[str, object]:
        H, r, k = self.horizon, self.r_dim, self.k_dim
        z_ref_np = z0 if z_tube_ref is None else z_tube_ref

        z0_f = np.asarray(z0, dtype=np.float32).reshape(r)
        ztgt_f = np.asarray(z_tgt, dtype=np.float32).reshape(r)
        zref_f = np.asarray(z_ref_np, dtype=np.float32).reshape(r)

        z_pred0 = self.Phi32 @ z0_f
        z_tgt_rep = np.tile(ztgt_f, H)
        two_g_qp = 2.0 * (self.Q_bar_Gamma_T32 @ (z_pred0 - z_tgt_rep))
        pz_ref = np.tile(self.Po32 @ zref_f, H)
        e0_perp = (self.Phi_perp32 @ z0_f) - pz_ref

        if u_init is not None:
            u_vec = np.clip(np.asarray(u_init, dtype=np.float32).reshape(H * k), 0.0, u_max).copy()
        else:
            u_vec = np.full(H * k, 0.10 * u_max, dtype=np.float32)

        lam = np.zeros(H, dtype=np.float32)
        rho = float(rho_init)
        eps_norm = np.float32(1e-8)
        ds32 = np.float32(delta_safe)
        umax32 = np.float32(u_max)

        for _ in range(outer_iters):
            step_size = np.float32(1.0 / (2.0 * self.L_qp + rho * self.gamma_perp_norm2 + 1e-6))
            y = u_vec.copy()
            theta = 1.0
            converged_inner = False
            for it in range(inner_iters):
                e_perp = (e0_perp + self.Gamma_perp32 @ y).reshape(H, r)
                d_norm = np.sqrt(np.sum(e_perp * e_perp, axis=1) + eps_norm)
                al_coeff = np.maximum(0.0, lam + np.float32(rho) * (d_norm - ds32)) / d_norm
                grad_e = (e_perp * al_coeff[:, None]).reshape(H * r)
                grad_u = self.two_H_qp32 @ y + two_g_qp + self.Gamma_perp_T32 @ grad_e

                u_next = np.clip(y - step_size * grad_u, np.float32(0.0), umax32)
                theta_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
                beta = np.float32((theta - 1.0) / theta_next)
                diff_u = u_next - u_vec
                y = u_next + beta * diff_u
                if tol > 0.0 and it >= 3:
                    rel_step = float(np.linalg.norm(diff_u)) / max(1.0, float(np.linalg.norm(u_next)))
                    if rel_step <= tol:
                        u_vec = u_next
                        converged_inner = True
                        break
                u_vec = u_next
                theta = theta_next

            e_perp_end = (e0_perp + self.Gamma_perp32 @ u_vec).reshape(H, r)
            d_end = np.sqrt(np.sum(e_perp_end * e_perp_end, axis=1) + eps_norm)
            viol_vec = d_end - ds32
            lam = np.maximum(np.float32(0.0), lam + np.float32(rho) * viol_vec)
            if converged_inner and float(np.max(viol_vec)) <= 1e-4:
                break
            rho *= rho_mult

        if enforce_feas_projection:
            e_check = (e0_perp + self.Gamma_perp32 @ u_vec).reshape(H, r)
            max_v = float(np.max(np.linalg.norm(e_check, axis=1)))
            if max_v > delta_safe:
                max_zero = float(np.max(np.linalg.norm(e0_perp.reshape(H, r), axis=1)))
                if max_zero <= delta_safe:
                    lo, hi = 0.0, 1.0
                    for _ in range(12):
                        mid = np.float32(0.5 * (lo + hi))
                        e_mid = (e0_perp + self.Gamma_perp32 @ (mid * u_vec)).reshape(H, r)
                        if float(np.max(np.linalg.norm(e_mid, axis=1))) <= delta_safe:
                            lo = float(mid)
                        else:
                            hi = float(mid)
                    u_vec = np.float32(lo) * u_vec

        z_flat = z_pred0 + self.Gamma32 @ u_vec
        e_final = (e0_perp + self.Gamma_perp32 @ u_vec).reshape(H, r)
        off_norms = np.linalg.norm(e_final, axis=1).astype(np.float64)
        dz_pred = (z_flat - z_tgt_rep).astype(np.float64)
        err0 = (z0_f - ztgt_f).astype(np.float64)
        u_arr = u_vec.reshape(H, k).astype(np.float64)
        lq_cost = float(
            self.q_weight * np.dot(err0, err0)
            + np.dot(dz_pred, self.Q_bar_np64 @ dz_pred)
            + self.r_weight * np.sum(u_arr * u_arr)
        )
        z_fut = z_flat.reshape(H, r).astype(np.float64)
        z_arr = np.vstack([np.asarray(z0, dtype=np.float64).reshape(1, r), z_fut])
        max_off = float(np.max(off_norms))
        viol_rate = float(np.mean(off_norms > delta_safe + 1e-6))
        sat_rate = float(1.0 - viol_rate)
        target_err = float(
            np.linalg.norm(z_arr[-1] - np.asarray(z_tgt, dtype=np.float64))
            / (np.linalg.norm(np.asarray(z0, dtype=np.float64) - np.asarray(z_tgt, dtype=np.float64)) + 1e-8)
        )
        total_energy = float(np.sum(u_arr ** 2))
        return {
            "u_traj": u_arr,
            "z_traj": z_arr,
            "off_norms": off_norms,
            "max_offtarget": max_off,
            "violation_rate": viol_rate,
            "constraint_sat_rate": sat_rate,
            "feasible": bool(max_off <= delta_safe + 1e-6),
            "relative_target_error": target_err,
            "control_energy": total_energy,
            "lq_cost": lq_cost,
        }

    @torch.no_grad()
    def solve(
        self,
        z0: np.ndarray,
        z_tgt: np.ndarray,
        delta_safe: float = 0.15,
        u_max: float = 1.0,
        z_tube_ref: Optional[np.ndarray] = None,
        u_init: Optional[np.ndarray] = None,
        outer_iters: int = 4,
        inner_iters: int = 25,
        rho_init: float = 80.0,
        rho_mult: float = 2.5,
        enforce_feas_projection: bool = True,
        tol: float = 1e-4,
    ) -> Dict[str, object]:
        """Solve box- and tube-constrained MPC via condensed tensorized FISTA-AL."""
        if self.backend == "numpy":
            return self._solve_numpy(
                z0, z_tgt, delta_safe, u_max, z_tube_ref, u_init,
                outer_iters, inner_iters, rho_init, rho_mult, enforce_feas_projection, tol,
            )

        H, r, k = self.horizon, self.r_dim, self.k_dim
        z_ref_np = z0 if z_tube_ref is None else z_tube_ref

        z0_t = torch.as_tensor(z0, dtype=self.dtype, device=self.device).reshape(r)
        ztgt_t = torch.as_tensor(z_tgt, dtype=self.dtype, device=self.device).reshape(r)
        zref_t = torch.as_tensor(z_ref_np, dtype=self.dtype, device=self.device).reshape(r)

        z_pred0 = self.Phi_t @ z0_t  # (Hr,)
        z_tgt_rep = ztgt_t.repeat(H)
        g_qp = self.Q_bar_Gamma_t.t() @ (z_pred0 - z_tgt_rep)  # (Hk,)

        pz_ref = (self.P_off_t @ zref_t).repeat(H)
        e0_perp = (self.Phi_perp_t @ z0_t) - pz_ref  # (Hr,)

        if u_init is not None:
            u_vec = torch.as_tensor(u_init, dtype=self.dtype, device=self.device).reshape(H * k).clone()
            u_vec.clamp_(0.0, u_max)
        else:
            u_vec = torch.full((H * k,), 0.10 * u_max, dtype=self.dtype, device=self.device)

        lam = torch.zeros(H, dtype=self.dtype, device=self.device)
        rho = float(rho_init)
        eps_norm = 1e-8
        two_H_qp = 2.0 * self.H_qp_t
        two_g_qp = 2.0 * g_qp

        for _ in range(outer_iters):
            step_size = 1.0 / (2.0 * self.L_qp + rho * self.gamma_perp_norm2 + 1e-6)
            y = u_vec.clone()
            theta = 1.0
            for _ in range(inner_iters):
                e_perp = (e0_perp + self.Gamma_perp_t @ y).view(H, r)
                d_norm = torch.sqrt(torch.sum(e_perp * e_perp, dim=1) + eps_norm)
                al_coeff = torch.clamp(lam + rho * (d_norm - delta_safe), min=0.0) / d_norm
                grad_e = (e_perp * al_coeff.unsqueeze(1)).reshape(H * r)
                grad_u = two_H_qp @ y + two_g_qp + self.Gamma_perp_t.t() @ grad_e

                u_next = torch.clamp(y - step_size * grad_u, min=0.0, max=u_max)
                theta_next = 0.5 * (1.0 + (1.0 + 4.0 * theta * theta) ** 0.5)
                beta = (theta - 1.0) / theta_next
                y = u_next + beta * (u_next - u_vec)
                u_vec = u_next
                theta = theta_next

            e_perp_end = (e0_perp + self.Gamma_perp_t @ u_vec).view(H, r)
            d_end = torch.sqrt(torch.sum(e_perp_end * e_perp_end, dim=1) + eps_norm)
            lam = torch.clamp(lam + rho * (d_end - delta_safe), min=0.0)
            rho *= rho_mult

        if enforce_feas_projection:
            e_check = (e0_perp + self.Gamma_perp_t @ u_vec).view(H, r)
            max_v = float(torch.max(torch.linalg.norm(e_check, dim=1)).item())
            if max_v > delta_safe:
                e_zero = e0_perp.view(H, r)
                max_zero = float(torch.max(torch.linalg.norm(e_zero, dim=1)).item())
                if max_zero <= delta_safe:
                    lo, hi = 0.0, 1.0
                    for _ in range(12):
                        mid = 0.5 * (lo + hi)
                        e_mid = (e0_perp + self.Gamma_perp_t @ (mid * u_vec)).view(H, r)
                        if float(torch.max(torch.linalg.norm(e_mid, dim=1)).item()) <= delta_safe:
                            lo = mid
                        else:
                            hi = mid
                    u_vec = lo * u_vec

        z_flat = z_pred0 + self.Gamma_t @ u_vec
        e_final = (e0_perp + self.Gamma_perp_t @ u_vec).view(H, r)
        off_norms_t = torch.linalg.norm(e_final, dim=1)
        dz_pred = z_flat - z_tgt_rep
        err0 = z0_t - ztgt_t
        lq_cost = float(
            self.q_weight * torch.dot(err0, err0)
            + torch.dot(dz_pred, self.Q_bar_t @ dz_pred)
            + self.r_weight * torch.dot(u_vec, u_vec)
        )

        u_arr = u_vec.view(H, k).detach().cpu().numpy().astype(np.float64)
        z_fut = z_flat.view(H, r).detach().cpu().numpy().astype(np.float64)
        z_arr = np.vstack([np.asarray(z0, dtype=np.float64).reshape(1, r), z_fut])
        off_norms = off_norms_t.detach().cpu().numpy().astype(np.float64)
        max_off = float(np.max(off_norms))
        viol_rate = float(np.mean(off_norms > delta_safe + 1e-6))
        sat_rate = float(1.0 - viol_rate)
        target_err = float(
            np.linalg.norm(z_arr[-1] - np.asarray(z_tgt, dtype=np.float64))
            / (np.linalg.norm(np.asarray(z0, dtype=np.float64) - np.asarray(z_tgt, dtype=np.float64)) + 1e-8)
        )
        total_energy = float(np.sum(u_arr ** 2))
        return {
            "u_traj": u_arr,
            "z_traj": z_arr,
            "off_norms": off_norms,
            "max_offtarget": max_off,
            "violation_rate": viol_rate,
            "constraint_sat_rate": sat_rate,
            "feasible": bool(max_off <= delta_safe + 1e-6),
            "relative_target_error": target_err,
            "control_energy": total_energy,
            "lq_cost": lq_cost,
        }

    @torch.no_grad()
    def solve_batch(
        self,
        z0_batch: np.ndarray,
        z_tgt_batch: np.ndarray,
        delta_safe: float = 0.15,
        u_max: float = 1.0,
        z_tube_ref_batch: Optional[np.ndarray] = None,
        u_init_batch: Optional[np.ndarray] = None,
        outer_iters: int = 4,
        inner_iters: int = 25,
        rho_init: float = 80.0,
        rho_mult: float = 2.5,
        enforce_feas_projection: bool = True,
    ) -> Dict[str, object]:
        """Solve B parallel MPC trajectories simultaneously via batched GPU/CPU tensor FISTA-AL."""
        if self.backend != "torch":
            raise RuntimeError("solve_batch requires backend='torch'.")
        B = int(z0_batch.shape[0])
        H, r, k = self.horizon, self.r_dim, self.k_dim
        z_ref_b = z0_batch if z_tube_ref_batch is None else z_tube_ref_batch

        z0_t = torch.as_tensor(z0_batch, dtype=self.dtype, device=self.device).reshape(B, r)
        ztgt_t = torch.as_tensor(z_tgt_batch, dtype=self.dtype, device=self.device).reshape(B, r)
        zref_t = torch.as_tensor(z_ref_b, dtype=self.dtype, device=self.device).reshape(B, r)

        z_pred0 = z0_t @ self.Phi_t.t()  # (B, Hr)
        z_tgt_rep = ztgt_t.repeat(1, H)  # (B, Hr)
        g_qp = (z_pred0 - z_tgt_rep) @ self.Q_bar_Gamma_t  # (B, Hk)

        pz_ref = (zref_t @ self.P_off_t.t()).repeat(1, H)
        e0_perp = z0_t @ self.Phi_perp_t.t() - pz_ref  # (B, Hr)

        if u_init_batch is not None:
            u_vec = torch.as_tensor(u_init_batch, dtype=self.dtype, device=self.device).reshape(B, H * k).clone()
            u_vec.clamp_(0.0, u_max)
        else:
            u_vec = torch.full((B, H * k), 0.10 * u_max, dtype=self.dtype, device=self.device)

        lam = torch.zeros((B, H), dtype=self.dtype, device=self.device)
        rho = float(rho_init)
        eps_norm = 1e-8
        two_H_qp_T = 2.0 * self.H_qp_t.t()
        two_g_qp = 2.0 * g_qp
        Gamma_perp_T = self.Gamma_perp_t.t()

        for _ in range(outer_iters):
            step_size = 1.0 / (2.0 * self.L_qp + rho * self.gamma_perp_norm2 + 1e-6)
            y = u_vec.clone()
            theta = 1.0
            for _ in range(inner_iters):
                e_perp = (e0_perp + y @ Gamma_perp_T).view(B, H, r)
                d_norm = torch.sqrt(torch.sum(e_perp * e_perp, dim=2) + eps_norm)
                al_coeff = torch.clamp(lam + rho * (d_norm - delta_safe), min=0.0) / d_norm
                grad_e = (e_perp * al_coeff.unsqueeze(2)).reshape(B, H * r)
                grad_u = y @ two_H_qp_T + two_g_qp + grad_e @ self.Gamma_perp_t

                u_next = torch.clamp(y - step_size * grad_u, min=0.0, max=u_max)
                theta_next = 0.5 * (1.0 + (1.0 + 4.0 * theta * theta) ** 0.5)
                beta = (theta - 1.0) / theta_next
                y = u_next + beta * (u_next - u_vec)
                u_vec = u_next
                theta = theta_next

            e_perp_end = (e0_perp + u_vec @ Gamma_perp_T).view(B, H, r)
            d_end = torch.sqrt(torch.sum(e_perp_end * e_perp_end, dim=2) + eps_norm)
            lam = torch.clamp(lam + rho * (d_end - delta_safe), min=0.0)
            rho *= rho_mult

        if enforce_feas_projection:
            e_check = (e0_perp + u_vec @ Gamma_perp_T).view(B, H, r)
            max_v = torch.max(torch.linalg.norm(e_check, dim=2), dim=1).values
            max_zero = torch.max(torch.linalg.norm(e0_perp.view(B, H, r), dim=2), dim=1).values
            need_bisect = (max_v > delta_safe) & (max_zero <= delta_safe)
            if bool(torch.any(need_bisect)):
                lo = torch.zeros(B, 1, dtype=self.dtype, device=self.device)
                hi = torch.ones(B, 1, device=self.device, dtype=self.dtype)
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    e_mid = (e0_perp + (mid * u_vec) @ Gamma_perp_T).view(B, H, r)
                    ok = torch.max(torch.linalg.norm(e_mid, dim=2), dim=1).values <= delta_safe
                    lo = torch.where(ok.unsqueeze(1), mid, lo)
                    hi = torch.where(ok.unsqueeze(1), hi, mid)
                scale = torch.where(need_bisect.unsqueeze(1), lo, torch.ones_like(lo))
                u_vec = scale * u_vec

        z_flat = z_pred0 + u_vec @ self.Gamma_t.t()
        e_final = (e0_perp + u_vec @ Gamma_perp_T).view(B, H, r)
        off_norms = torch.linalg.norm(e_final, dim=2)  # (B, H)
        max_off = torch.max(off_norms, dim=1).values
        viol_rate = torch.mean((off_norms > delta_safe + 1e-6).to(self.dtype), dim=1)
        dz_pred = z_flat - z_tgt_rep
        err0 = z0_t - ztgt_t
        lq_costs = (
            self.q_weight * torch.sum(err0 * err0, dim=1)
            + torch.sum(dz_pred * (dz_pred @ self.Q_bar_t.t()), dim=1)
            + self.r_weight * torch.sum(u_vec * u_vec, dim=1)
        )
        z_end = z_flat.view(B, H, r)[:, -1, :]
        rel_err = torch.linalg.norm(z_end - ztgt_t, dim=1) / (torch.linalg.norm(err0, dim=1) + 1e-8)

        return {
            "u_traj": u_vec.view(B, H, k).detach().cpu().numpy().astype(np.float64),
            "max_offtarget": max_off.detach().cpu().numpy().astype(np.float64),
            "constraint_sat_rate": float((1.0 - viol_rate).mean().item()),
            "feasible_rate": float((max_off <= delta_safe + 1e-6).to(self.dtype).mean().item()),
            "relative_target_error": float(rel_err.mean().item()),
            "control_energy": float(torch.sum(u_vec * u_vec, dim=1).mean().item()),
            "lq_cost": float(lq_costs.mean().item()),
        }


class TubeSafeMPCController:
    """Delay-compensated, RPI tube-tightened receding-horizon controller with ancillary DARE feedback."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        P_offtarget: np.ndarray,
        horizon: int = 8,
        q_weight: float = 1.0,
        r_weight: float = 0.05,
        q_anc: float = 2.0,
        r_anc: float = 0.1,
        device: Optional[str] = None,
        backend: str = "numpy",
    ) -> None:
        self.solver = CondensedSafeMPCSolver(
            A, B, P_offtarget, horizon=horizon, q_weight=q_weight, r_weight=r_weight, device=device, backend=backend
        )
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.P_off = np.asarray(P_offtarget, dtype=np.float64)
        self.horizon = int(horizon)
        r, k = self.B.shape
        Q_anc = float(q_anc) * np.eye(r, dtype=np.float64)
        R_anc = float(r_anc) * np.eye(k, dtype=np.float64)
        try:
            P_anc = solve_discrete_are(self.A, self.B, Q_anc, R_anc)
        except Exception:
            P_anc = 5.0 * Q_anc
        self.K_anc = np.linalg.solve(R_anc + self.B.T @ P_anc @ self.B, self.B.T @ P_anc @ self.A)
        self.gamma_cl = float(np.linalg.norm(self.A - self.B @ self.K_anc, ord=2))

    def estimate_rpi_margin(
        self,
        w_bound: float,
        delay_steps: int = 0,
        max_frac: float = 0.45,
        delta_safe: float = 0.25,
    ) -> float:
        """Compute theoretical Robust Positive Invariant (RPI) tube tightening margin from w_bound and gamma_cl."""
        gamma_eff = min(0.96, float(self.gamma_cl))
        raw_rpi = float(w_bound) / max(0.04, 1.0 - gamma_eff) + float(delay_steps) * float(w_bound)
        return float(min(float(max_frac) * float(delta_safe), max(0.0, raw_rpi)))

    def step(
        self,
        z_meas: np.ndarray,
        z_tgt: np.ndarray,
        z_ref: np.ndarray,
        delta_safe: float,
        u_max: float,
        delay_buffer: Optional[List[np.ndarray]] = None,
        tube_margin: float = 0.0,
        u_warm: Optional[np.ndarray] = None,
        z_nom_prev: Optional[np.ndarray] = None,
        use_ancillary: bool = True,
        use_safety_guard: bool = True,
    ) -> Dict[str, object]:
        """Advance one receding-horizon control step with delay compensation and tightened tube."""
        z_hat = np.asarray(z_meas, dtype=np.float64).copy()
        if delay_buffer:
            for u_pend in delay_buffer:
                z_hat = self.A @ z_hat + self.B @ np.asarray(u_pend, dtype=np.float64)

        # Compute unforced baseline drift to keep tightened nominal problem feasible whenever delta_safe permits
        z_unforced = z_hat.copy()
        min_unforced = 0.0
        for _ in range(self.horizon):
            z_unforced = self.A @ z_unforced
            min_unforced = max(min_unforced, float(np.linalg.norm(self.P_off @ (z_unforced - z_ref))))
        delta_min = min(float(delta_safe), min_unforced + 1e-4)
        delta_tight = max(delta_min, float(delta_safe) - float(tube_margin))

        nom_res = self.solver.solve(
            z0=z_hat,
            z_tgt=z_tgt,
            delta_safe=delta_tight,
            u_max=u_max,
            z_tube_ref=z_ref,
            u_init=u_warm,
            enforce_feas_projection=True,
        )
        u_nom = nom_res["u_traj"]
        v0 = np.clip(u_nom[0], 0.0, u_max)

        if use_ancillary and z_nom_prev is not None:
            err_tube = z_hat - np.asarray(z_nom_prev, dtype=np.float64)
            u_cand = np.clip(v0 - self.K_anc @ err_tube, 0.0, u_max)
            # 1-step nominal safety guard: prevent ancillary overshoot from breaching delta_safe
            if (not use_safety_guard) or float(np.linalg.norm(self.P_off @ (self.A @ z_hat + self.B @ u_cand - z_ref))) <= delta_safe + 1e-6:
                u_apply = u_cand
            else:
                u_apply = v0
        else:
            u_apply = v0

        z_nom_next = self.A @ z_hat + self.B @ u_apply
        u_warm_next = np.vstack([u_nom[1:], np.zeros((1, self.B.shape[1]), dtype=np.float64)])
        return {
            "u_apply": u_apply,
            "u_warm_next": u_warm_next,
            "z_nom_next": z_nom_next,
            "z_hat": z_hat,
            "delta_tight": delta_tight,
            "feasible": bool(nom_res["feasible"]),
            "nominal_max_off": float(nom_res["max_offtarget"]),
        }
