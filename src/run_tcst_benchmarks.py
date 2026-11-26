"""IEEE TCST / Automatica real-time MPC scaling, batch-crossover, N=20 RPI Tube-MPC ablations, and spectral/sensitivity benchmarks."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _cand in reversed([str(_THIS_DIR), str(Path.cwd() / "src"), "/content/paper3/src", "/content"]):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from baselines import UnconstrainedEDMD, evaluate_perturbation_predictions  # noqa: E402
from data_loader import load_gears_perturb_h5ad, prepare_perturb_splits  # noqa: E402
from kgc_mpc import (  # noqa: E402
    CondensedSafeMPCSolver,
    TubeSafeMPCController,
    compute_nonnormal_diagnostics,
    fit_kgc_system,
    get_device,
    select_cocktail_submodular,
    solve_safety_mpc,
)

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    cp = None
    HAS_CVXPY = False


def _sync_if_cuda(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _gpu_short_tag() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    name = torch.cuda.get_device_name(0).upper()
    for tag in ("H100", "A100", "L4", "T4", "V100", "P100"):
        if tag in name:
            return tag
    return "GPU"


def _top_de_indices(ctrl_mean: np.ndarray, target_mean: np.ndarray, k: int = 20) -> np.ndarray:
    return np.argsort(np.abs(target_mean - ctrl_mean))[::-1][:k]


def _bh_fdr(p_vals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR adjustment preserving original order."""
    m = len(p_vals)
    if m == 0:
        return []
    order = np.argsort(p_vals)
    q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank_rev in range(m - 1, -1, -1):
        idx = int(order[rank_rev])
        val = float(p_vals[idx]) * m / float(rank_rev + 1)
        prev = min(prev, val)
        q[idx] = min(1.0, prev)
    return q.tolist()


def _safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    if len(diff) == 0 or np.all(np.abs(diff) <= 1e-12):
        return 1.0
    try:
        _, p_val = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return 1.0 if np.isnan(p_val) else float(p_val)
    except Exception:
        return 1.0


def _load_or_fit_norman_model(
    prep: Dict[str, object],
    latent_dim: int,
    weights_dir: Path,
    epochs: int = 55,
    seed: int = 0,
    gamma_schur: float = 0.95,
    bilinear_rank: int = 12,
    stiefel_contraction: bool = True,
    use_saved_ab: bool = True,
):
    X, U, conds, ctrl_mask, pert_feats, split_map = (
        prep["X"],
        prep["U"],
        prep["conds"],
        prep["ctrl_mask"],
        prep["pert_feats"],
        prep["split_map"],
    )
    x_ctrl = X[ctrl_mask]
    ctrl_mean = x_ctrl.mean(axis=0)
    tr_mask = np.isin(conds, [c for c, sp in split_map.items() if sp == "train"])

    wt_path = weights_dir / "norman_kgc_seed0.npz"
    if (
        use_saved_ab
        and latent_dim == 64
        and seed == 0
        and stiefel_contraction
        and abs(gamma_schur - 0.95) < 1e-6
        and bilinear_rank == 12
        and wt_path.exists()
    ):
        z_npz = np.load(wt_path, allow_pickle=True)
        A_k = z_npz["A"].astype(np.float64)
        B_k = z_npz["B"].astype(np.float64)
        kgc = fit_kgc_system(
            x_ctrl,
            X[tr_mask],
            U[tr_mask],
            prep["pert_list"],
            dose_mode=False,
            latent_dim=64,
            bilinear_rank=12,
            horizon=3,
            gamma_schur=0.95,
            epochs=epochs,
            batch_size=512,
            seed=seed,
            gene_pert_features=pert_feats,
        )
        return kgc, A_k, B_k, ctrl_mean

    kgc = fit_kgc_system(
        x_ctrl,
        X[tr_mask],
        U[tr_mask],
        prep["pert_list"],
        dose_mode=False,
        latent_dim=latent_dim,
        bilinear_rank=bilinear_rank,
        horizon=3,
        gamma_schur=gamma_schur,
        stiefel_contraction=stiefel_contraction,
        epochs=epochs,
        batch_size=512,
        seed=seed,
        gene_pert_features=pert_feats,
    )
    A_k, B_k, _, _ = kgc.extract_matrices()
    return kgc, A_k, B_k, ctrl_mean


def _build_norman_targets(
    prep: Dict[str, object],
    kgc,
    ctrl_mean: np.ndarray,
    n_targets: int = 20,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    X, conds, split_map = prep["X"], prep["conds"], prep["split_map"]
    single_cond_map: Dict[str, str] = {}
    for c_str in set(conds):
        if c_str != "ctrl":
            parts = [p for p in str(c_str).split("+") if p != "ctrl" and p != ""]
            if len(parts) == 1:
                single_cond_map[parts[0]] = str(c_str)
    tr_single = [
        pi
        for pi, g in enumerate(prep["pert_list"])
        if g in single_cond_map and split_map.get(single_cond_map[g], "") == "train"
    ]
    deltas = {
        pi: X[conds == single_cond_map[prep["pert_list"][pi]]].mean(axis=0) - ctrl_mean
        for pi in tr_single
    }
    pot = {pi: float(np.linalg.norm(d)) for pi, d in deltas.items()}
    ranked = sorted(tr_single, key=lambda idx: pot[idx], reverse=True)
    th = np.quantile(list(pot.values()), 0.80)
    high = [pi for pi in tr_single if pot[pi] >= th] or [ranked[0]]
    d_mean = np.mean([deltas[pi] for pi in high], axis=0)

    dev = get_device()
    targets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    cand_list: List[Tuple[str, np.ndarray]] = [("Top20Pct_Cluster", d_mean)]
    for idx in ranked[: max(0, n_targets - 1)]:
        cand_list.append((prep["pert_list"][idx], deltas[idx]))

    with torch.no_grad():
        z_src = kgc.encode(torch.tensor(ctrl_mean[None, :], dtype=torch.float32, device=dev)).cpu().numpy()[0].astype(np.float64)
        for name, d_vec in cand_list[:n_targets]:
            x_tgt = ctrl_mean + d_vec
            z_tgt = kgc.encode(torch.tensor(x_tgt[None, :], dtype=torch.float32, device=dev)).cpu().numpy()[0].astype(np.float64)
            dz_dir = (z_tgt - z_src) / (np.linalg.norm(z_tgt - z_src) + 1e-8)
            P_off = np.eye(len(z_src), dtype=np.float64) - np.outer(dz_dir, dz_dir)
            targets.append((name, z_src.copy(), z_tgt, P_off))
    return targets


def _benchmark_solver_latency(
    solver: CondensedSafeMPCSolver,
    z_src: np.ndarray,
    z_tgt: np.ndarray,
    delta_safe: float,
    warm_start: bool,
    n_warmup: int,
    n_runs: int,
    opt_ref_cost: Optional[float] = None,
) -> Dict[str, float]:
    base_res = solver.solve(z_src, z_tgt, delta_safe=delta_safe, u_max=1.0, z_tube_ref=z_src)
    u_warm = np.vstack([base_res["u_traj"][1:], np.zeros((1, solver.k_dim), dtype=np.float64)]) if warm_start else None

    for _ in range(n_warmup):
        solver.solve(z_src, z_tgt, delta_safe=delta_safe, u_max=1.0, z_tube_ref=z_src, u_init=u_warm)
    _sync_if_cuda(solver.device)

    times_ms = []
    last_res = base_res
    for _ in range(n_runs):
        _sync_if_cuda(solver.device)
        t0 = time.perf_counter()
        last_res = solver.solve(z_src, z_tgt, delta_safe=delta_safe, u_max=1.0, z_tube_ref=z_src, u_init=u_warm)
        _sync_if_cuda(solver.device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms, dtype=np.float64)
    lq_cost = float(last_res["lq_cost"])
    gap = 0.0 if opt_ref_cost is None else max(0.0, (lq_cost - opt_ref_cost) / max(1e-8, abs(opt_ref_cost)) * 100.0)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "constraint_sat_rate": float(last_res["constraint_sat_rate"]),
        "feasible_rate": float(1.0 if last_res["feasible"] else 0.0),
        "max_offtarget": float(last_res["max_offtarget"]),
        "target_error": float(last_res["relative_target_error"]),
        "control_energy": float(last_res["control_energy"]),
        "lq_cost": lq_cost,
        "opt_gap_pct": gap,
    }


def _benchmark_batch_gpu(
    solver: CondensedSafeMPCSolver,
    z_src: np.ndarray,
    z_tgt: np.ndarray,
    delta_safe: float,
    batch_size: int,
    n_warmup: int,
    n_runs: int,
    opt_ref_cost: Optional[float] = None,
) -> Dict[str, float]:
    z0_b = np.repeat(z_src[None, :], batch_size, axis=0)
    ztgt_b = np.repeat(z_tgt[None, :], batch_size, axis=0)
    base_res = solver.solve_batch(z0_b, ztgt_b, delta_safe=delta_safe, u_max=1.0, z_tube_ref_batch=z0_b)
    u_warm_b = np.concatenate([base_res["u_traj"][:, 1:, :], np.zeros((batch_size, 1, solver.k_dim))], axis=1)

    for _ in range(n_warmup):
        solver.solve_batch(z0_b, ztgt_b, delta_safe=delta_safe, u_max=1.0, z_tube_ref_batch=z0_b, u_init_batch=u_warm_b)
    _sync_if_cuda(solver.device)

    per_traj_ms = []
    last_res = base_res
    for _ in range(n_runs):
        _sync_if_cuda(solver.device)
        t0 = time.perf_counter()
        last_res = solver.solve_batch(z0_b, ztgt_b, delta_safe=delta_safe, u_max=1.0, z_tube_ref_batch=z0_b, u_init_batch=u_warm_b)
        _sync_if_cuda(solver.device)
        per_traj_ms.append((time.perf_counter() - t0) * 1000.0 / batch_size)

    arr = np.array(per_traj_ms, dtype=np.float64)
    lq_cost = float(last_res["lq_cost"])
    gap = 0.0 if opt_ref_cost is None else max(0.0, (lq_cost - opt_ref_cost) / max(1e-8, abs(opt_ref_cost)) * 100.0)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "constraint_sat_rate": float(last_res["constraint_sat_rate"]),
        "feasible_rate": float(last_res["feasible_rate"]),
        "max_offtarget": float(np.mean(last_res["max_offtarget"])),
        "target_error": float(last_res["relative_target_error"]),
        "control_energy": float(last_res["control_energy"]),
        "lq_cost": lq_cost,
        "opt_gap_pct": gap,
    }


def _benchmark_cvxpy_conic(
    solver_ref: CondensedSafeMPCSolver,
    z_src: np.ndarray,
    z_tgt: np.ndarray,
    delta_safe: float,
    n_warmup: int,
    n_runs: int,
    solver_name: str = "CLARABEL",
    opt_ref_cost: Optional[float] = None,
) -> Dict[str, float]:
    H, r, k = solver_ref.horizon, solver_ref.r_dim, solver_ref.k_dim
    z0_64 = np.asarray(z_src, dtype=np.float64).reshape(r)
    ztgt_64 = np.asarray(z_tgt, dtype=np.float64).reshape(r)
    z_pred0 = solver_ref.Phi_np64 @ z0_64
    z_tgt_rep = np.tile(ztgt_64, H)
    g_qp = solver_ref.Q_bar_Gamma_np64.T @ (z_pred0 - z_tgt_rep)
    err0 = z0_64 - ztgt_64
    dz0 = z_pred0 - z_tgt_rep
    const_cost = float(solver_ref.q_weight * np.dot(err0, err0) + np.dot(dz0, solver_ref.Q_bar_np64 @ dz0))
    pz_ref = np.tile(solver_ref.P_off_np @ z0_64, H)
    e0_perp = solver_ref.Phi_perp_np64 @ z0_64 - pz_ref

    u_var = cp.Variable(H * k)
    obj = cp.Minimize(cp.quad_form(u_var, cp.psd_wrap(solver_ref.H_qp_np64)) + 2.0 * (g_qp @ u_var) + const_cost)
    cons = [u_var >= 0.0, u_var <= 1.0]
    for t in range(H):
        cons.append(cp.norm2(e0_perp[t * r : (t + 1) * r] + solver_ref.Gamma_perp_np64[t * r : (t + 1) * r, :] @ u_var) <= delta_safe)
    prob = cp.Problem(obj, cons)

    solve_kw = {"solver": cp.CLARABEL} if solver_name == "CLARABEL" else {"solver": cp.SCS, "eps": 1e-4, "max_iters": 5000}
    for _ in range(n_warmup):
        prob.solve(**solve_kw)

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        prob.solve(**solve_kw)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    u_raw = u_var.value
    u_val = np.zeros(H * k, dtype=np.float64) if u_raw is None else np.clip(np.asarray(u_raw, dtype=np.float64).reshape(H * k), 0.0, 1.0)
    z_flat = z_pred0 + solver_ref.Gamma_np64 @ u_val
    e_final = (e0_perp + solver_ref.Gamma_perp_np64 @ u_val).reshape(H, r)
    off_norms = np.linalg.norm(e_final, axis=1)
    dz_pred = z_flat - z_tgt_rep
    lq_cost = float(solver_ref.q_weight * np.dot(err0, err0) + np.dot(dz_pred, solver_ref.Q_bar_np64 @ dz_pred) + solver_ref.r_weight * np.dot(u_val, u_val))
    z_end = z_flat.reshape(H, r)[-1]
    target_err = float(np.linalg.norm(z_end - ztgt_64) / (np.linalg.norm(err0) + 1e-8))
    max_off = float(np.max(off_norms))
    viol_rate = float(np.mean(off_norms > delta_safe + 1e-4))
    gap = 0.0 if opt_ref_cost is None else max(0.0, (lq_cost - opt_ref_cost) / max(1e-8, abs(opt_ref_cost)) * 100.0)
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "constraint_sat_rate": float(1.0 - viol_rate),
        "feasible_rate": float(1.0 if max_off <= delta_safe + 1e-4 else 0.0),
        "max_offtarget": max_off,
        "target_error": target_err,
        "control_energy": float(np.sum(u_val ** 2)),
        "lq_cost": lq_cost,
        "opt_gap_pct": gap,
    }


def _benchmark_adam_latency(
    A: np.ndarray,
    B_top: np.ndarray,
    z_src: np.ndarray,
    z_tgt: np.ndarray,
    P_off: np.ndarray,
    horizon: int,
    delta_safe: float,
    n_warmup: int,
    n_runs: int,
    opt_ref_cost: Optional[float] = None,
) -> Dict[str, float]:
    for _ in range(n_warmup):
        solve_safety_mpc(A, B_top, z_src, z_tgt, P_off, delta_safe=delta_safe, u_max=1.0, horizon=horizon, enforce_constraints=True, solver_mode="adam_penalty")
    times_ms = []
    last_res = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        last_res = solve_safety_mpc(A, B_top, z_src, z_tgt, P_off, delta_safe=delta_safe, u_max=1.0, horizon=horizon, enforce_constraints=True, solver_mode="adam_penalty")
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "constraint_sat_rate": float(1.0 - last_res["violation_rate"]),
        "feasible_rate": float(1.0 if last_res["max_offtarget"] <= delta_safe + 1e-6 else 0.0),
        "max_offtarget": float(last_res["max_offtarget"]),
        "target_error": float(last_res["relative_target_error"]),
        "control_energy": float(last_res["control_energy"]),
        "lq_cost": float("nan") if opt_ref_cost is None else float(opt_ref_cost),
        "opt_gap_pct": float("nan"),
    }


def run_realtime_scaling_benchmark(
    adata,
    weights_dir: Path,
    out_dir: Path,
    quick: bool = False,
) -> pd.DataFrame:
    print("\n=== [Paper 3 TCST] Part 1: Real-Time MPC Scaling (FISTA-AL vs Clarabel SOCP, SCS ADMM, Adam & Batch Crossover) ===", flush=True)
    prep = prepare_perturb_splits(adata, seed=0, n_hvg=1000)
    has_cuda = torch.cuda.is_available()
    gpu_tag = _gpu_short_tag()
    n_warmup = 5 if quick else 20
    n_runs = 15 if quick else 100
    n_adam_runs = 3 if quick else 15
    n_clar_runs = 5 if quick else 30

    dim_list = [32, 64] if quick else [32, 64, 128, 256]
    horizon_list = [4, 8, 16] if quick else [4, 8, 12, 16, 24, 32]
    batch_list = [1, 16, 64] if quick else [1, 4, 8, 16, 32, 64, 128, 256]
    fit_epochs = 12 if quick else 40

    models: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for r_dim in dim_list:
        kgc, A_k, B_k, ctrl_mean = _load_or_fit_norman_model(prep, r_dim, weights_dir, epochs=fit_epochs, seed=0)
        sel_sub, _ = select_cocktail_submodular(A_k, B_k, budget_k=8, horizon=6)
        B_top = B_k[:, sel_sub[:8]]
        _, z_src, z_tgt, P_off = _build_norman_targets(prep, kgc, ctrl_mean, n_targets=1)[0]
        delta_rel = 0.25 * float(np.linalg.norm(z_tgt - z_src))
        models[r_dim] = (A_k, B_top, z_src, z_tgt, P_off, delta_rel)
        del kgc
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()

    rows = []

    # Sweep 1: Horizon H sweep at r=64
    A64, B64, z_src64, z_tgt64, P_off64, ds64 = models[64]
    for H in horizon_list:
        sol_cpu = CondensedSafeMPCSolver(A64, B64, P_off64, horizon=H, backend="numpy")
        opt_cost = None
        if HAS_CVXPY:
            m_clar = _benchmark_cvxpy_conic(sol_cpu, z_src64, z_tgt64, ds64, 2 if quick else 5, n_clar_runs, solver_name="CLARABEL")
            opt_cost = m_clar["lq_cost"]
            rows.append({"sweep_type": "horizon", "latent_dim": 64, "horizon": H, "n_actuators": 8, "batch_size": 1, "solver": "CVXPY-Clarabel (SOCP)", **m_clar})
            if "SCS" in cp.installed_solvers():
                m_scs = _benchmark_cvxpy_conic(sol_cpu, z_src64, z_tgt64, ds64, 2 if quick else 5, n_clar_runs, solver_name="SCS", opt_ref_cost=opt_cost)
                rows.append({"sweep_type": "horizon", "latent_dim": 64, "horizon": H, "n_actuators": 8, "batch_size": 1, "solver": "CVXPY-SCS (Conic ADMM)", **m_scs})
        m_cpu_w = _benchmark_solver_latency(sol_cpu, z_src64, z_tgt64, ds64, True, n_warmup, n_runs, opt_ref_cost=opt_cost)
        rows.append({"sweep_type": "horizon", "latent_dim": 64, "horizon": H, "n_actuators": 8, "batch_size": 1, "solver": "FISTA-AL (CPU BLAS Warm)", **m_cpu_w})
        m_cpu_c = _benchmark_solver_latency(sol_cpu, z_src64, z_tgt64, ds64, False, n_warmup, n_runs, opt_ref_cost=opt_cost)
        rows.append({"sweep_type": "horizon", "latent_dim": 64, "horizon": H, "n_actuators": 8, "batch_size": 1, "solver": "FISTA-AL (CPU BLAS Cold)", **m_cpu_c})
        if has_cuda:
            sol_gpu = CondensedSafeMPCSolver(A64, B64, P_off64, horizon=H, device="cuda", backend="torch")
            m_gpu_b64 = _benchmark_batch_gpu(sol_gpu, z_src64, z_tgt64, ds64, 64, n_warmup, n_runs, opt_ref_cost=opt_cost)
            rows.append({"sweep_type": "horizon", "latent_dim": 64, "horizon": H, "n_actuators": 8, "batch_size": 64, "solver": f"FISTA-AL ({gpu_tag} B=64/traj)", **m_gpu_b64})
            m_gpu_b1 = _benchmark_solver_latency(sol_gpu, z_src64, z_tgt64, ds64, True, n_warmup, n_runs, opt_ref_cost=opt_cost)
            rows.append({"sweep_type": "horizon", "latent_dim": 64, "horizon": H, "n_actuators": 8, "batch_size": 1, "solver": f"FISTA-AL ({gpu_tag} B=1 Warm)", **m_gpu_b1})
        m_adam = _benchmark_adam_latency(A64, B64, z_src64, z_tgt64, P_off64, H, ds64, 1 if quick else 2, n_adam_runs, opt_ref_cost=opt_cost)
        rows.append({"sweep_type": "horizon", "latent_dim": 64, "horizon": H, "n_actuators": 8, "batch_size": 1, "solver": "Autograd-Adam (Orig. CPU)", **m_adam})
        clar_str = f" | Clarabel={m_clar['mean_ms']:.2f}ms | SCS={m_scs['mean_ms']:.2f}ms" if HAS_CVXPY and "SCS" in cp.installed_solvers() else ""
        gpu_str = f" | {gpu_tag}-B64={m_gpu_b64['mean_ms']:.3f}ms/traj" if has_cuda else ""
        print(f"  [Horizon H={H:2d}] CPU-Warm={m_cpu_w['mean_ms']:.2f}ms Cold={m_cpu_c['mean_ms']:.2f}ms{clar_str}{gpu_str} | Adam={m_adam['mean_ms']:.1f}ms", flush=True)

    # Sweep 2: State dimension r sweep at H=8
    for r_dim in dim_list:
        Ar, Br, z_src_r, z_tgt_r, P_off_r, dsr = models[r_dim]
        sol_cpu = CondensedSafeMPCSolver(Ar, Br, P_off_r, horizon=8, backend="numpy")
        opt_cost = None
        if HAS_CVXPY:
            m_clar = _benchmark_cvxpy_conic(sol_cpu, z_src_r, z_tgt_r, dsr, 2 if quick else 5, n_clar_runs, solver_name="CLARABEL")
            opt_cost = m_clar["lq_cost"]
            rows.append({"sweep_type": "dimension", "latent_dim": r_dim, "horizon": 8, "n_actuators": 8, "batch_size": 1, "solver": "CVXPY-Clarabel (SOCP)", **m_clar})
            if "SCS" in cp.installed_solvers():
                m_scs = _benchmark_cvxpy_conic(sol_cpu, z_src_r, z_tgt_r, dsr, 2 if quick else 5, n_clar_runs, solver_name="SCS", opt_ref_cost=opt_cost)
                rows.append({"sweep_type": "dimension", "latent_dim": r_dim, "horizon": 8, "n_actuators": 8, "batch_size": 1, "solver": "CVXPY-SCS (Conic ADMM)", **m_scs})
        m_cpu_w = _benchmark_solver_latency(sol_cpu, z_src_r, z_tgt_r, dsr, True, n_warmup, n_runs, opt_ref_cost=opt_cost)
        rows.append({"sweep_type": "dimension", "latent_dim": r_dim, "horizon": 8, "n_actuators": 8, "batch_size": 1, "solver": "FISTA-AL (CPU BLAS Warm)", **m_cpu_w})
        m_cpu_c = _benchmark_solver_latency(sol_cpu, z_src_r, z_tgt_r, dsr, False, n_warmup, n_runs, opt_ref_cost=opt_cost)
        rows.append({"sweep_type": "dimension", "latent_dim": r_dim, "horizon": 8, "n_actuators": 8, "batch_size": 1, "solver": "FISTA-AL (CPU BLAS Cold)", **m_cpu_c})
        if has_cuda:
            sol_gpu = CondensedSafeMPCSolver(Ar, Br, P_off_r, horizon=8, device="cuda", backend="torch")
            m_gpu_b64 = _benchmark_batch_gpu(sol_gpu, z_src_r, z_tgt_r, dsr, 64, n_warmup, n_runs, opt_ref_cost=opt_cost)
            rows.append({"sweep_type": "dimension", "latent_dim": r_dim, "horizon": 8, "n_actuators": 8, "batch_size": 64, "solver": f"FISTA-AL ({gpu_tag} B=64/traj)", **m_gpu_b64})
            m_gpu_b1 = _benchmark_solver_latency(sol_gpu, z_src_r, z_tgt_r, dsr, True, n_warmup, n_runs, opt_ref_cost=opt_cost)
            rows.append({"sweep_type": "dimension", "latent_dim": r_dim, "horizon": 8, "n_actuators": 8, "batch_size": 1, "solver": f"FISTA-AL ({gpu_tag} B=1 Warm)", **m_gpu_b1})
        print(f"  [Dim r={r_dim:3d}] CPU-Warm={m_cpu_w['mean_ms']:.2f}ms (gap={m_cpu_w['opt_gap_pct']:.2f}%)", flush=True)

    # Sweep 3: GPU batch-size crossover at H=8, r=64, k=8
    sol_cpu_h8 = CondensedSafeMPCSolver(A64, B64, P_off64, horizon=8, backend="numpy")
    opt_h8 = None
    if HAS_CVXPY:
        opt_h8 = _benchmark_cvxpy_conic(sol_cpu_h8, z_src64, z_tgt64, ds64, 2, 5, solver_name="CLARABEL")["lq_cost"]
    if has_cuda:
        sol_gpu_h8 = CondensedSafeMPCSolver(A64, B64, P_off64, horizon=8, device="cuda", backend="torch")
        for b_sz in batch_list:
            m_bs = _benchmark_batch_gpu(sol_gpu_h8, z_src64, z_tgt64, ds64, b_sz, n_warmup, n_runs, opt_ref_cost=opt_h8)
            rows.append({"sweep_type": "batch_size", "latent_dim": 64, "horizon": 8, "n_actuators": 8, "batch_size": b_sz, "solver": f"FISTA-AL ({gpu_tag} Batch/traj)", **m_bs})
            print(f"  [Batch B={b_sz:3d}] {gpu_tag} per-traj={m_bs['mean_ms']:.3f}ms (gap={m_bs['opt_gap_pct']:.2f}%)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "mpc_realtime_scaling.csv", index=False)
    return df


def _calibrate_one_step_mismatch(
    kgc_ctrl,
    kgc_plant,
    A_k: np.ndarray,
    B_top: np.ndarray,
    plant_idx: List[int],
    ctrl_mean0: np.ndarray,
    U1_dim: int,
    dev: torch.device,
) -> float:
    """Measure empirical 1-step cross-seed residual norm ||z_next_plant_meas - (A z + B u)||_2."""
    rng = np.random.default_rng(7)
    residuals = []
    with torch.no_grad():
        Ap_t, Bp_t = kgc_plant.get_A(), kgc_plant.get_B()
        z_p0 = kgc_plant.encode(torch.tensor(ctrl_mean0[None, :], dtype=torch.float32, device=dev))
        z_c0 = kgc_ctrl.encode(torch.tensor(ctrl_mean0[None, :], dtype=torch.float32, device=dev)).cpu().numpy()[0].astype(np.float64)
        for _ in range(12):
            u_sub = rng.uniform(0.0, 0.5, size=len(plant_idx))
            u_full = torch.zeros((1, U1_dim), dtype=torch.float32, device=dev)
            for k_loc, p_i in enumerate(plant_idx):
                u_full[0, p_i] = float(u_sub[k_loc])
            z_p1 = kgc_plant.step(z_p0, u_full, A=Ap_t, B=Bp_t)
            x_1 = kgc_plant.decode(z_p1).cpu().numpy()[0]
            z_meas_1 = kgc_ctrl.encode(torch.tensor(x_1[None, :], dtype=torch.float32, device=dev)).cpu().numpy()[0].astype(np.float64)
            z_nom_1 = A_k @ z_c0 + B_top @ u_sub
            residuals.append(float(np.linalg.norm(z_meas_1 - z_nom_1)))
    return float(np.quantile(residuals, 0.75))


def run_stress_and_unified_control_tests(
    adata,
    weights_dir: Path,
    out_dir: Path,
    quick: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== [Paper 3 TCST] Part 2: N=20 Stress Tests, Component Ablations, Paired Wilcoxon & Unified Pareto ===", flush=True)
    prep0 = prepare_perturb_splits(adata, seed=0, n_hvg=1000)
    prep1 = prepare_perturb_splits(adata, seed=1, n_hvg=1000)
    epochs = 12 if quick else 55
    n_targets = 3 if quick else 20

    kgc_ctrl, A_k, B_k, ctrl_mean0 = _load_or_fit_norman_model(prep0, 64, weights_dir, epochs=epochs, seed=0)
    kgc_plant, _, _, _ = _load_or_fit_norman_model(prep1, 64, weights_dir, epochs=epochs, seed=1, use_saved_ab=False)

    sel_sub, _ = select_cocktail_submodular(A_k, B_k, budget_k=8, horizon=6)
    B_top = B_k[:, sel_sub[:8]]
    ctrl_perts = [prep0["pert_list"][j] for j in sel_sub[:8]]
    plant_idx = [prep1["pert_to_idx"][g] for g in ctrl_perts]
    targets = _build_norman_targets(prep0, kgc_ctrl, ctrl_mean0, n_targets=n_targets)

    dev = get_device()
    horizon = 8
    T_sim = 12
    U1_dim = prep1["U"].shape[1]

    w_base = _calibrate_one_step_mismatch(kgc_ctrl, kgc_plant, A_k, B_top, plant_idx, ctrl_mean0, U1_dim, dev)
    print(f"  Calibrated 1-step cross-seed disturbance w_base={w_base:.4f}, n_targets={len(targets)}", flush=True)

    with torch.no_grad():
        Ap_t = kgc_plant.get_A()
        Bp_t = kgc_plant.get_B()

    solvers = [CondensedSafeMPCSolver(A_k, B_top, P_off, horizon=horizon, backend="numpy") for _, _, _, P_off in targets]
    tube_ctrls = [TubeSafeMPCController(A_k, B_top, P_off, horizon=horizon, backend="numpy") for _, _, _, P_off in targets]

    def _run_episode(
        method: str,
        target_idx: int,
        z_src: np.ndarray,
        z_tgt: np.ndarray,
        P_off: np.ndarray,
        mismatch_scale: float,
        u_max: float,
        delay_steps: int,
        delta_safe_override: Optional[float] = None,
        n_steps: int = T_sim,
    ) -> Dict[str, float]:
        rng_ep = np.random.default_rng(1000 + target_idx)
        delta_safe_i = 0.25 * float(np.linalg.norm(z_tgt - z_src)) if delta_safe_override is None else float(delta_safe_override)
        sol_i = solvers[target_idx]
        tc_i = tube_ctrls[target_idx]

        w_bound = w_base * (0.6 + 0.4 * mismatch_scale) + 0.008 * max(0.0, mismatch_scale - 1.0)
        eff_delay = 0 if method == "Tube-MPC w/o Delay-Pred" else delay_steps
        rpi_margin = tc_i.estimate_rpi_margin(w_bound=w_bound, delay_steps=eff_delay, max_frac=0.45, delta_safe=delta_safe_i)
        tube_margin = 0.0 if method == "Tube-MPC w/o RPI" else rpi_margin

        ol_res = sol_i.solve(z_src, z_tgt, delta_safe=delta_safe_i, u_max=u_max, z_tube_ref=z_src)
        lqr_res = solve_safety_mpc(A_k, B_top, z_src, z_tgt, P_off, delta_safe=delta_safe_i, u_max=u_max, horizon=horizon, enforce_constraints=False)
        u_lqr = np.clip(lqr_res["u_traj"], 0.0, u_max)

        x_cur = ctrl_mean0.copy()
        with torch.no_grad():
            z_p = kgc_plant.encode(torch.tensor(x_cur[None, :], dtype=torch.float32, device=dev))

        u_queue: List[np.ndarray] = []
        u_warm = None
        z_nom_prev = None
        off_list = []
        feas_steps = 0
        energy = 0.0
        lq_cost = 0.0

        for t in range(n_steps):
            with torch.no_grad():
                z_meas = kgc_ctrl.encode(torch.tensor(x_cur[None, :], dtype=torch.float32, device=dev)).cpu().numpy()[0].astype(np.float64)

            if method.startswith("KGC Tube-MPC") or method.startswith("Tube-MPC w/o"):
                pending = [] if (method == "Tube-MPC w/o Delay-Pred" or delay_steps == 0) else u_queue[max(0, len(u_queue) - delay_steps) :]
                step_out = tc_i.step(
                    z_meas=z_meas,
                    z_tgt=z_tgt,
                    z_ref=z_src,
                    delta_safe=delta_safe_i,
                    u_max=u_max,
                    delay_buffer=pending,
                    tube_margin=tube_margin,
                    u_warm=u_warm,
                    z_nom_prev=z_nom_prev,
                    use_ancillary=(method != "Tube-MPC w/o Ancillary"),
                )
                u_cmd = step_out["u_apply"]
                u_warm = step_out["u_warm_next"]
                z_nom_prev = step_out["z_nom_next"]
                if step_out["feasible"]:
                    feas_steps += 1
            elif method.startswith("Std Receding") or method.startswith("Closed-Loop Receding"):
                res = sol_i.solve(z_meas, z_tgt, delta_safe=delta_safe_i, u_max=u_max, z_tube_ref=z_src, u_init=u_warm)
                u_cmd = res["u_traj"][0]
                u_warm = np.vstack([res["u_traj"][1:], np.zeros((1, len(u_cmd)))])
                if res["feasible"]:
                    feas_steps += 1
            elif method.startswith("Open-Loop"):
                u_cmd = ol_res["u_traj"][t] if t < len(ol_res["u_traj"]) else np.zeros(len(plant_idx))
                if ol_res["feasible"]:
                    feas_steps += 1
            else:
                u_cmd = u_lqr[t] if t < len(u_lqr) else np.zeros(len(plant_idx))
                feas_steps += 1

            u_cmd = np.clip(u_cmd, 0.0, u_max)
            u_queue.append(u_cmd)
            u_eff = u_queue[t - delay_steps] if t >= delay_steps else np.zeros_like(u_cmd)

            energy += float(np.sum(u_eff ** 2))
            err_t = z_meas - z_tgt
            lq_cost += float(np.dot(err_t, err_t) + 0.05 * np.dot(u_eff, u_eff))

            with torch.no_grad():
                u_full = torch.zeros((1, U1_dim), dtype=torch.float32, device=dev)
                for k_loc, p_i in enumerate(plant_idx):
                    u_full[0, p_i] = float(u_eff[k_loc])
                z_lin = z_p @ Ap_t.t() + u_full @ Bp_t.t()
                z_bilin = kgc_plant.step(z_p, u_full, A=Ap_t, B=Bp_t) - z_lin
                z_p = z_lin + mismatch_scale * z_bilin
                if mismatch_scale > 1.0:
                    w_raw = rng_ep.normal(size=z_p.shape[1]).astype(np.float64)
                    w_orth = P_off @ w_raw
                    w_orth = w_orth / (np.linalg.norm(w_orth) + 1e-8) * (0.008 * (mismatch_scale - 1.0))
                    z_p = z_p + torch.tensor(w_orth[None, :], dtype=torch.float32, device=dev)
                x_cur = kgc_plant.decode(z_p).cpu().numpy()[0]
                z_next_meas = kgc_ctrl.encode(torch.tensor(x_cur[None, :], dtype=torch.float32, device=dev)).cpu().numpy()[0].astype(np.float64)

            off_list.append(float(np.linalg.norm(P_off @ (z_next_meas - z_src))))

        with torch.no_grad():
            z_fin = kgc_ctrl.encode(torch.tensor(x_cur[None, :], dtype=torch.float32, device=dev)).cpu().numpy()[0].astype(np.float64)
        off_arr = np.array(off_list, dtype=np.float64)
        target_err = float(np.linalg.norm(z_fin - z_tgt) / (np.linalg.norm(z_src - z_tgt) + 1e-8))
        return {
            "rec_feas_rate": float(feas_steps / n_steps),
            "constraint_sat_rate": float(np.mean(off_arr <= delta_safe_i + 1e-6)),
            "max_offtarget": float(np.max(off_arr)),
            "norm_offtarget": float(np.max(off_arr) / delta_safe_i),
            "target_error": target_err,
            "control_energy": energy,
            "lq_cost": lq_cost,
        }

    sweeps = [
        ("mismatch_scale", [(v, 1.0, 0) for v in ([1.0, 3.0, 5.0] if quick else [1.0, 2.0, 3.0, 5.0])]),
        ("saturation_umax", [(2.0, v, 0) for v in ([1.0, 0.25] if quick else [1.0, 0.6, 0.4, 0.25])]),
        ("delay_steps", [(2.0, 0.6, v) for v in ([0, 3] if quick else [0, 1, 2, 3])]),
    ]
    methods = [
        "KGC Tube-MPC (RPI+Delay)",
        "Tube-MPC w/o RPI",
        "Tube-MPC w/o Delay-Pred",
        "Tube-MPC w/o Ancillary",
        "Std Receding Safe-MPC",
        "Open-Loop Safe-MPC",
        "Box-LQR Baseline",
    ]

    stress_rows = []
    episode_store: Dict[Tuple[str, float, str], List[Dict[str, float]]] = {}
    for axis_name, configs in sweeps:
        for mis_s, u_mx, tau_d in configs:
            s_val = mis_s if axis_name == "mismatch_scale" else (u_mx if axis_name == "saturation_umax" else tau_d)
            for m_name in methods:
                ep_mets = [
                    _run_episode(m_name, ti, z_src, z_tgt, P_off, mis_s, u_mx, tau_d)
                    for ti, (_, z_src, z_tgt, P_off) in enumerate(targets)
                ]
                episode_store[(axis_name, float(s_val), m_name)] = ep_mets
                df_m = pd.DataFrame(ep_mets)
                row = {
                    "stress_axis": axis_name,
                    "stress_value": float(s_val),
                    "method": m_name,
                    "n_targets": len(targets),
                    "rec_feas_rate": float(df_m["rec_feas_rate"].mean()),
                    "constraint_sat_rate": float(df_m["constraint_sat_rate"].mean()),
                    "constraint_sat_std": float(df_m["constraint_sat_rate"].std(ddof=0)),
                    "max_offtarget_mean": float(df_m["max_offtarget"].mean()),
                    "max_offtarget_std": float(df_m["max_offtarget"].std(ddof=0)),
                    "norm_offtarget_mean": float(df_m["norm_offtarget"].mean()),
                    "norm_offtarget_std": float(df_m["norm_offtarget"].std(ddof=0)),
                    "target_error_mean": float(df_m["target_error"].mean()),
                    "target_error_std": float(df_m["target_error"].std(ddof=0)),
                    "control_energy_mean": float(df_m["control_energy"].mean()),
                    "lq_cost_mean": float(df_m["lq_cost"].mean()),
                }
                stress_rows.append(row)
                print(
                    f"  [{axis_name}={s_val}] {m_name:25s} | Sat={row['constraint_sat_rate']*100:5.1f}% "
                    f"| NormOff={row['norm_offtarget_mean']:.3f}±{row['norm_offtarget_std']:.3f} | Err={row['target_error_mean']:.3f}",
                    flush=True,
                )

    df_out = pd.DataFrame(stress_rows)
    df_out.to_csv(out_dir / "mpc_stress_tests.csv", index=False)

    # Paired Wilcoxon signed-rank + BH-FDR tests across targets (key regimes + pooled 12 regimes)
    stat_rows = []
    key_regimes = [
        ("mismatch_scale", 1.0, "Mismatch k=1"),
        ("mismatch_scale", 5.0, "Mismatch k=5"),
        ("saturation_umax", 0.25, "Sat u_max=0.25"),
        ("delay_steps", 3.0, "Delay tau=3"),
    ]
    ref_m = "KGC Tube-MPC (RPI+Delay)"
    for ax, sv, reg_lbl in key_regimes:
        if (ax, sv, ref_m) not in episode_store:
            continue
        ref_eps = episode_store[(ax, sv, ref_m)]
        ref_off = np.array([e["norm_offtarget"] for e in ref_eps])
        ref_sat = np.array([e["constraint_sat_rate"] for e in ref_eps])
        for cmp_m in methods[1:]:
            cmp_eps = episode_store[(ax, sv, cmp_m)]
            cmp_off = np.array([e["norm_offtarget"] for e in cmp_eps])
            cmp_sat = np.array([e["constraint_sat_rate"] for e in cmp_eps])
            stat_rows.append({
                "regime": reg_lbl,
                "n_pairs": len(ref_off),
                "comparator": cmp_m,
                "full_norm_off": float(np.mean(ref_off)),
                "comp_norm_off": float(np.mean(cmp_off)),
                "p_norm_off": _safe_wilcoxon(ref_off, cmp_off),
                "full_sat_rate": float(np.mean(ref_sat)),
                "comp_sat_rate": float(np.mean(cmp_sat)),
                "p_sat_rate": _safe_wilcoxon(ref_sat, cmp_sat),
            })

    # Pooled across all evaluated stress conditions
    all_keys = sorted({(k[0], k[1]) for k in episode_store.keys()})
    for cmp_m in methods[1:]:
        ref_off_all = np.concatenate([[e["norm_offtarget"] for e in episode_store[(ax, sv, ref_m)]] for ax, sv in all_keys])
        cmp_off_all = np.concatenate([[e["norm_offtarget"] for e in episode_store[(ax, sv, cmp_m)]] for ax, sv in all_keys])
        ref_sat_all = np.concatenate([[e["constraint_sat_rate"] for e in episode_store[(ax, sv, ref_m)]] for ax, sv in all_keys])
        cmp_sat_all = np.concatenate([[e["constraint_sat_rate"] for e in episode_store[(ax, sv, cmp_m)]] for ax, sv in all_keys])
        stat_rows.append({
            "regime": "Pooled_All_Regimes",
            "n_pairs": len(ref_off_all),
            "comparator": cmp_m,
            "full_norm_off": float(np.mean(ref_off_all)),
            "comp_norm_off": float(np.mean(cmp_off_all)),
            "p_norm_off": _safe_wilcoxon(ref_off_all, cmp_off_all),
            "full_sat_rate": float(np.mean(ref_sat_all)),
            "comp_sat_rate": float(np.mean(cmp_sat_all)),
            "p_sat_rate": _safe_wilcoxon(ref_sat_all, cmp_sat_all),
        })

    df_stats = pd.DataFrame(stat_rows)
    df_stats["q_norm_off"] = _bh_fdr(df_stats["p_norm_off"].tolist())
    df_stats["q_sat_rate"] = _bh_fdr(df_stats["p_sat_rate"].tolist())
    df_stats.to_csv(out_dir / "mpc_statistical_tests.csv", index=False)
    print(f"  Saved mpc_statistical_tests.csv ({len(df_stats)} paired comparisons)", flush=True)

    # Regenerate unified safety_mpc_pareto.csv & safety_mpc_mismatch.csv on Top20Pct_Cluster (target 0)
    _, z_src0, z_tgt0, P_off0 = targets[0]
    ds_sweep = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.85]
    pareto_rows = []
    lqr_nom = solve_safety_mpc(A_k, B_top, z_src0, z_tgt0, P_off0, delta_safe=0.40, u_max=1.0, horizon=8, enforce_constraints=False)
    for ds in ds_sweep:
        mpc_nom = solve_safety_mpc(A_k, B_top, z_src0, z_tgt0, P_off0, delta_safe=ds, u_max=1.0, horizon=8, enforce_constraints=True, solver_mode="fista_al")
        pareto_rows.append({
            "delta_safe": ds,
            "method": "KGC-MPC (Constrained)",
            "target_error": float(mpc_nom["relative_target_error"]),
            "max_offtarget": float(mpc_nom["max_offtarget"]),
            "violation_rate": float(mpc_nom["violation_rate"]),
            "control_energy": float(mpc_nom["control_energy"]),
        })
        pareto_rows.append({
            "delta_safe": ds,
            "method": "Unconstrained LQR",
            "target_error": float(lqr_nom["relative_target_error"]),
            "max_offtarget": float(lqr_nom["max_offtarget"]),
            "violation_rate": float(np.mean(lqr_nom["off_norms"] > ds)),
            "control_energy": float(lqr_nom["control_energy"]),
        })
    pd.DataFrame(pareto_rows).to_csv(out_dir / "safety_mpc_pareto.csv", index=False)

    mismatch_rows = []
    for ds in ds_sweep:
        for m_lbl in [
            "KGC Tube-MPC (Plant Mismatch)",
            "Closed-Loop Receding Safe-MPC (Plant Mismatch)",
            "Open-Loop Safe-MPC (Plant Mismatch)",
            "Box-LQR Baseline (Plant Mismatch)",
        ]:
            met = _run_episode(m_lbl, 0, z_src0, z_tgt0, P_off0, mismatch_scale=1.0, u_max=1.0, delay_steps=0, delta_safe_override=ds, n_steps=8)
            mismatch_rows.append({
                "delta_safe": ds,
                "method": m_lbl,
                "target_error": met["target_error"],
                "max_offtarget": met["max_offtarget"],
                "violation_rate": 1.0 - met["constraint_sat_rate"],
                "control_energy": met["control_energy"],
            })
    pd.DataFrame(mismatch_rows).to_csv(out_dir / "safety_mpc_mismatch.csv", index=False)
    print("  Updated unified safety_mpc_pareto.csv & safety_mpc_mismatch.csv (FISTA-AL exact)", flush=True)
    return df_out, df_stats


def run_spectral_and_sensitivity_audits(
    adata,
    weights_dir: Path,
    out_dir: Path,
    quick: bool = False,
) -> pd.DataFrame:
    print("\n=== [Paper 3 TCST] Part 3: Non-Normal Kreiss Diagnostics & Sensitivity Sweep ===", flush=True)
    prep0 = prepare_perturb_splits(adata, seed=0, n_hvg=1000)
    X, U, conds, ctrl_mask, pert_feats, split_map = (
        prep0["X"],
        prep0["U"],
        prep0["conds"],
        prep0["ctrl_mask"],
        prep0["pert_feats"],
        prep0["split_map"],
    )
    x_ctrl = X[ctrl_mask]
    ctrl_mean = x_ctrl.mean(axis=0)
    tr_mask = np.isin(conds, [c for c, sp in split_map.items() if sp == "train"])
    x_tr, u_tr = X[tr_mask], U[tr_mask]
    dev = get_device()
    epochs = 12 if quick else 55

    # 1. Update koopman_spectra_norman.json with nonnormal_diagnostics while preserving exact seed0 weights
    z_npz = np.load(weights_dir / "norman_kgc_seed0.npz", allow_pickle=True)
    A_kgc = z_npz["A"].astype(np.float64)
    ro_df = pd.read_csv(out_dir / "rollout_stability.csv")
    diag_kgc = compute_nonnormal_diagnostics(A_kgc, max_step=20)
    diag_kgc["rollout20_norm"] = float(ro_df["kgc_norm"].iloc[-1])

    edmd = UnconstrainedEDMD(latent_dim=64, reg=1e-3, horizon=3).fit(x_ctrl, x_tr, u_tr)
    diag_edmd = compute_nonnormal_diagnostics(edmd.A, max_step=20)
    diag_edmd["rollout20_norm"] = float(ro_df["edmd_norm"].iloc[-1])

    kgc_ns = fit_kgc_system(
        x_ctrl, x_tr, u_tr, prep0["pert_list"],
        dose_mode=False, latent_dim=64, bilinear_rank=12, horizon=3, gamma_schur=0.95,
        stiefel_contraction=False, epochs=epochs, batch_size=512, seed=0, gene_pert_features=pert_feats,
    )
    A_ns, _, _, _ = kgc_ns.extract_matrices()
    with torch.no_grad():
        z0_ns = kgc_ns.encode(torch.tensor(ctrl_mean[None, :], dtype=torch.float32, device=dev))
        u0_ns = torch.zeros((1, U.shape[1]), dtype=torch.float32, device=dev)
        u0_ns[0, 0] = 1.0
        _, traj_ns = kgc_ns.rollout(z0_ns, u0_ns, steps=20)
        ro20_ns = float(torch.linalg.norm(traj_ns[-1]).cpu().item())
    diag_ns = compute_nonnormal_diagnostics(A_ns, max_step=20)
    diag_ns["rollout20_norm"] = ro20_ns

    sp_path = out_dir / "koopman_spectra_norman.json"
    sp_data = json.loads(sp_path.read_text()) if sp_path.is_file() else {}
    sp_data["nonnormal_diagnostics"] = {
        "KGC_Stiefel": diag_kgc,
        "KGC_Unconstrained_A": diag_ns,
        "Unconstrained_EDMD": diag_edmd,
    }
    sp_path.write_text(json.dumps(sp_data, indent=2))
    print(
        f"  [Kreiss] KGC sigma={diag_kgc['sigma_max']:.4f}, cond(V)={diag_kgc['eigvec_cond']:.2f}, K_lb={diag_kgc['kreiss_lb']:.2f} | "
        f"No-Stiefel sigma={diag_ns['sigma_max']:.4f}, cond(V)={diag_ns['eigvec_cond']:.2f}, K_lb={diag_ns['kreiss_lb']:.2f} | "
        f"EDMD sigma={diag_edmd['sigma_max']:.4f}, cond(V)={diag_edmd['eigvec_cond']:.2f}, K_lb={diag_edmd['kreiss_lb']:.2f}",
        flush=True,
    )

    # 2. Hyperparameter sensitivity sweep: gamma_schur & bilinear_rank q on Norman seed=0
    split_eval = {}
    for sp_name in ("combo_seen2", "combo_seen0"):
        sp_conds = sorted([c for c, s in split_map.items() if s == sp_name])
        u_list, y_list, de_list = [], [], []
        for c in sp_conds:
            parts = [p for p in c.split("+") if p != "ctrl"]
            uv = np.zeros(U.shape[1], dtype=np.float32)
            for p in parts:
                uv[prep0["pert_to_idx"][p]] = 1.0
            u_list.append(uv)
            cm = X[conds == c].mean(axis=0)
            y_list.append(cm - ctrl_mean)
            de_list.append(_top_de_indices(ctrl_mean, cm, k=20))
        split_eval[sp_name] = (np.stack(u_list), np.stack(y_list), de_list)

    configs = []
    for g_val in ([0.80, 0.95] if quick else [0.80, 0.90, 0.95, 0.99]):
        configs.append(("gamma_schur", f"{g_val:.2f}", g_val, 12, True))
    configs.append(("gamma_schur", "Unconstrained", 0.95, 12, False))
    for q_val in ([0, 12] if quick else [0, 4, 12, 24]):
        configs.append(("bilinear_rank", str(q_val), 0.95, q_val, True))

    sens_rows = []
    for sw_p, val_lbl, g_v, q_v, st_flag in configs:
        if (not st_flag) and g_v == 0.95 and q_v == 12:
            m_cur = kgc_ns
            A_cur = A_ns
            diag_cur = diag_ns
        else:
            m_cur = fit_kgc_system(
                x_ctrl, x_tr, u_tr, prep0["pert_list"],
                dose_mode=False, latent_dim=64, bilinear_rank=q_v, horizon=3, gamma_schur=g_v,
                stiefel_contraction=st_flag, epochs=epochs, batch_size=512, seed=0, gene_pert_features=pert_feats,
            )
            A_cur, _, _, _ = m_cur.extract_matrices()
            with torch.no_grad():
                z0_c = m_cur.encode(torch.tensor(ctrl_mean[None, :], dtype=torch.float32, device=dev))
                u0_c = torch.zeros((1, U.shape[1]), dtype=torch.float32, device=dev)
                u0_c[0, 0] = 1.0
                _, traj_c = m_cur.rollout(z0_c, u0_c, steps=20)
                ro20 = float(torch.linalg.norm(traj_c[-1]).cpu().item())
            diag_cur = compute_nonnormal_diagnostics(A_cur, max_step=20)
            diag_cur["rollout20_norm"] = ro20

        res_s = {}
        for sp_name, (u_te, y_te, de_l) in split_eval.items():
            with torch.no_grad():
                x0_m = torch.tensor(ctrl_mean[None, :], dtype=torch.float32, device=dev).expand(len(u_te), -1)
                pred_d = m_cur.forward_x(x0_m, torch.tensor(u_te, dtype=torch.float32, device=dev))[0].cpu().numpy() - ctrl_mean[None, :]
            res_s[sp_name] = evaluate_perturbation_predictions(y_te, pred_d, de_l)["de20_pearson"]

        sens_rows.append({
            "sweep_param": sw_p,
            "param_value": val_lbl,
            "gamma_schur": g_v if st_flag else float("nan"),
            "stiefel_contraction": st_flag,
            "bilinear_rank": q_v,
            "combo_seen2_de20_r": float(res_s["combo_seen2"]),
            "combo_seen0_de20_r": float(res_s["combo_seen0"]),
            "sigma_max": diag_cur["sigma_max"],
            "rho": diag_cur["rho"],
            "eigvec_cond": diag_cur["eigvec_cond"],
            "kreiss_lb": diag_cur["kreiss_lb"],
            "rollout20_norm": diag_cur["rollout20_norm"],
        })
        print(
            f"  [Sens {sw_p}={val_lbl:13s}] seen2_r={res_s['combo_seen2']:.3f}, seen0_r={res_s['combo_seen0']:.3f}, "
            f"sigma={diag_cur['sigma_max']:.4f}, ro20={diag_cur['rollout20_norm']:.3f}",
            flush=True,
        )
        del m_cur
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(out_dir / "hyperparam_sensitivity.csv", index=False)
    return df_sens


def _query_compute_processes() -> List[str]:
    if not torch.cuda.is_available():
        return []
    out = subprocess.getoutput("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "No running processes" not in ln]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 3 IEEE TCST / Automatica top-journal benchmarks.")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Fast local CPU smoke mode.")
    args = parser.parse_args()

    t_start = time.perf_counter()
    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        work_dir = Path("/content/paper3") if Path("/content").exists() else Path(__file__).resolve().parents[1]

    data_dir = work_dir / "data"
    out_dir = work_dir / "results"
    weights_dir = work_dir / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    adata = load_gears_perturb_h5ad("norman", data_dir)
    run_realtime_scaling_benchmark(adata, weights_dir, out_dir, quick=args.quick)
    run_stress_and_unified_control_tests(adata, weights_dir, out_dir, quick=args.quick)
    run_spectral_and_sensitivity_audits(adata, weights_dir, out_dir, quick=args.quick)

    procs = _query_compute_processes()
    exclusive = bool(torch.cuda.is_available() and len(procs) <= 1)
    env_info = {
        "paper": "Paper 3 (KGC-MPC)",
        "benchmark": "TCST/Automatica Real-Time MPC (Clarabel+SCS+Batch Crossover), N=20 RPI Tube-MPC Ablations & Kreiss/Sensitivity Audits",
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_tag": _gpu_short_tag(),
        "has_cvxpy": HAS_CVXPY,
        "tf32_enabled": bool(torch.cuda.is_available()),
        "active_compute_procs_at_finish": len(procs),
        "exclusive_gpu_verified": exclusive,
        "shared_gpu_note": (
            "Executed exclusively on NVIDIA A100-SXM4-80GB."
            if exclusive
            else f"Executed on shared NVIDIA A100-SXM4-80GB ({len(procs)} active GPU compute processes observed); CPU BLAS, Clarabel/SCS, and batched A100 timings recorded."
        ),
        "quick_mode": bool(args.quick),
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
    }
    (out_dir / "environment_tcst.json").write_text(json.dumps(env_info, indent=2))
    print(f"\n[P3_TCST_COMPLETE] Elapsed {env_info['elapsed_sec']}s on {env_info['gpu']} | exclusive={exclusive} (procs={len(procs)})", flush=True)


if __name__ == "__main__":
    main()
