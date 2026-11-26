"""Offline smoke tests for Paper 3 (KGC-MPC): Schur stability, bilinear epistasis, Gramian prioritization, and Safety-MPC."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import (  # noqa: E402
    UnconstrainedEDMD,
    compute_linear_additive_baseline,
    evaluate_perturbation_predictions,
    fit_and_predict_cpa,
    fit_and_predict_gears_gnn,
)
from kgc_mpc import (  # noqa: E402
    CondensedSafeMPCSolver,
    TubeSafeMPCController,
    compute_controllability_gramian,
    compute_nonnormal_diagnostics,
    fit_kgc_system,
    score_targets_gramian,
    select_cocktail_submodular,
    solve_safety_mpc,
)


def main() -> None:
    rng = np.random.default_rng(42)
    n_genes = 60
    n_perts = 8
    n_ctrl = 120

    # Synthetic nonlinear dynamical regime with bilinear interaction on pert 0 + pert 1
    x_ctrl = rng.normal(0.0, 0.3, size=(n_ctrl, n_genes)).astype(np.float32)
    B_true = rng.normal(0.0, 0.8, size=(n_perts, n_genes)).astype(np.float32)
    pert_feats = B_true[:, :24] / (np.linalg.norm(B_true[:, :24], axis=1, keepdims=True) + 1e-6)

    x_list, u_list = [], []
    # Single perturbations 0..5
    for p in range(6):
        u = np.zeros(n_perts, dtype=np.float32)
        u[p] = 1.0
        for _ in range(30):
            x = rng.normal(0.0, 0.25, size=n_genes).astype(np.float32) + B_true[p]
            x_list.append(x)
            u_list.append(u)
    # Combo perturbations (0+1, 2+3) with positive synergy
    for p1, p2 in [(0, 1), (2, 3)]:
        u = np.zeros(n_perts, dtype=np.float32)
        u[p1] = 1.0
        u[p2] = 1.0
        for _ in range(30):
            x = rng.normal(0.0, 0.25, size=n_genes).astype(np.float32) + B_true[p1] + B_true[p2] + 0.4 * (B_true[p1] * B_true[p2])
            x_list.append(x)
            u_list.append(u)

    x_pert = np.stack(x_list, axis=0)
    u_pert = np.stack(u_list, axis=0)
    pert_names = [f"G{i}" for i in range(n_perts)]

    model = fit_kgc_system(
        x_ctrl=x_ctrl,
        x_pert=x_pert,
        u_pert=u_pert,
        pert_names=pert_names,
        dose_mode=False,
        latent_dim=24,
        bilinear_rank=6,
        horizon=3,
        gamma_schur=0.95,
        epochs=15,
        batch_size=64,
        seed=42,
        gene_pert_features=pert_feats,
        device="cpu",
    )
    rho_A = model.spectral_radius()
    assert rho_A < 0.96, f"Schur stability violated: rho(A)={rho_A}"

    # Unseen combo (0+2)
    u_test = np.zeros((2, n_perts), dtype=np.float32)
    u_test[0, [0, 2]] = 1.0
    u_test[1, [1, 3]] = 1.0
    with torch.no_grad():
        x0_m = torch.tensor(x_ctrl.mean(axis=0, keepdims=True)).expand(2, -1)
        z0_zero = model.encode(x0_m[:1])
        assert float(torch.linalg.norm(z0_zero)) < 1e-4, f"Zero-preserving encode failed: norm={float(torch.linalg.norm(z0_zero))}"
        x_hat, _, _ = model.forward_x(x0_m, torch.tensor(u_test))
    assert x_hat.shape == (2, n_genes)

    # Gramian prioritization & submodular cocktail selection
    A, B, C, _ = model.extract_matrices()
    W_c = compute_controllability_gramian(A, B, horizon=5)
    assert np.all(np.linalg.eigvalsh(W_c) > 0)
    z0 = np.zeros(A.shape[0])
    ztgt = B[:, 0] * 1.5
    scores, energies = score_targets_gramian(A, B, z0, ztgt, horizon=5)
    assert int(np.argmax(scores)) == 0, f"Expected actuator 0 highest, got {np.argmax(scores)}"

    sel, gains = select_cocktail_submodular(A, B, budget_k=3, horizon=5)
    assert len(sel) == 3 and gains[0] >= gains[1] >= gains[2] > 0, f"Submodular diminishing returns failed: {gains}"

    # Safety-constrained MPC vs Unconstrained LQR (unified exact FISTA-AL solver)
    d_unit = ztgt / (np.linalg.norm(ztgt) + 1e-8)
    P_off = np.eye(A.shape[0]) - np.outer(d_unit, d_unit)
    res_mpc = solve_safety_mpc(A, B, z0, ztgt, P_off, delta_safe=0.15, u_max=1.0, horizon=6, enforce_constraints=True, solver_mode="fista_al")
    res_lqr = solve_safety_mpc(A, B, z0, ztgt, P_off, delta_safe=0.15, u_max=1.0, horizon=6, enforce_constraints=False)
    assert res_mpc["violation_rate"] == 0.0, f"Unified FISTA-AL solve_safety_mpc breached tube: {res_mpc['violation_rate']}"
    assert res_mpc["max_offtarget"] <= res_lqr["max_offtarget"] + 0.05

    diag_kgc = compute_nonnormal_diagnostics(A, max_step=10)
    assert diag_kgc["sigma_max"] <= 0.96 and diag_kgc["kreiss_lb"] >= 1.0 and np.isfinite(diag_kgc["eigvec_cond"])

    model_nostiefel = fit_kgc_system(
        x_ctrl=x_ctrl, x_pert=x_pert, u_pert=u_pert, pert_names=pert_names,
        dose_mode=False, latent_dim=24, bilinear_rank=6, horizon=3, gamma_schur=0.95,
        stiefel_contraction=False, epochs=4, batch_size=64, seed=42, gene_pert_features=pert_feats, device="cpu",
    )
    assert model_nostiefel.A_unconstrained is not None and model_nostiefel.get_A().shape == (24, 24)

    # Condensed FISTA-AL solver (torch + numpy BLAS + batch) & Tube-MPC delay/RPI tightening check
    fista_solver = CondensedSafeMPCSolver(A, B, P_off, horizon=6, device="cpu", backend="numpy")
    res_fista = fista_solver.solve(z0, ztgt, delta_safe=0.15, u_max=1.0, z_tube_ref=z0)
    assert res_fista["feasible"] and res_fista["constraint_sat_rate"] == 1.0
    u_warm = np.vstack([res_fista["u_traj"][1:], np.zeros((1, B.shape[1]))])
    res_warm = fista_solver.solve(z0, ztgt, delta_safe=0.15, u_max=1.0, z_tube_ref=z0, u_init=u_warm)
    assert res_warm["feasible"]

    fista_torch = CondensedSafeMPCSolver(A, B, P_off, horizon=6, device="cpu", backend="torch")
    res_batch = fista_torch.solve_batch(np.repeat(z0[None, :], 4, axis=0), np.repeat(ztgt[None, :], 4, axis=0), delta_safe=0.15, u_max=1.0)
    assert res_batch["feasible_rate"] == 1.0 and res_batch["constraint_sat_rate"] == 1.0

    tube_ctrl = TubeSafeMPCController(A, B, P_off, horizon=6, device="cpu", backend="numpy")
    rpi_m = tube_ctrl.estimate_rpi_margin(w_bound=0.01, delay_steps=1, delta_safe=0.15)
    assert 0.0 < rpi_m <= 0.45 * 0.15 + 1e-8
    t_step = tube_ctrl.step(
        z_meas=z0, z_tgt=ztgt, z_ref=z0, delta_safe=0.15, u_max=0.6,
        delay_buffer=[np.full(B.shape[1], 0.1)], tube_margin=rpi_m, u_warm=u_warm, z_nom_prev=z0,
    )
    assert t_step["feasible"] and np.all(t_step["u_apply"] >= 0.0) and np.all(t_step["u_apply"] <= 0.6 + 1e-8)
    t_step_noanc = tube_ctrl.step(
        z_meas=z0, z_tgt=ztgt, z_ref=z0, delta_safe=0.15, u_max=0.6,
        delay_buffer=[], tube_margin=rpi_m, u_warm=u_warm, z_nom_prev=z0, use_ancillary=False,
    )
    assert t_step_noanc["feasible"] and np.all(t_step_noanc["u_apply"] <= 0.6 + 1e-8)

    # Baselines sanity + Small-Gain Bilinear bound check
    model_sg = fit_kgc_system(
        x_ctrl=x_ctrl, x_pert=x_pert, u_pert=u_pert, pert_names=pert_names,
        dose_mode=False, latent_dim=24, bilinear_rank=6, horizon=3, gamma_schur=0.95,
        bilin_max_gain=0.04, epochs=5, batch_size=64, seed=42, gene_pert_features=pert_feats, device="cpu",
    )
    with torch.no_grad():
        x_sg, zT_sg, _ = model_sg.forward_x(torch.tensor(x_ctrl[:2]), torch.tensor(u_test), steps=10)
    assert torch.all(torch.isfinite(zT_sg)) and float(torch.linalg.norm(zT_sg)) < 100.0

    edmd = UnconstrainedEDMD(latent_dim=24).fit(x_ctrl, x_pert, u_pert)
    d_edmd = edmd.predict_delta(u_test)
    d_cpa = fit_and_predict_cpa(x_ctrl, x_pert, u_pert, u_test, epochs=10, device="cpu")
    d_gears = fit_and_predict_gears_gnn(x_ctrl, x_pert, u_pert, u_test, pert_feats, epochs=10, device="cpu")
    assert d_edmd.shape == d_cpa.shape == d_gears.shape == (2, n_genes)
    met = evaluate_perturbation_predictions(d_edmd, d_cpa, [np.arange(20), np.arange(20)])
    assert np.isfinite(met["de20_pearson"])

    print(
        f"[Paper 3 Smoke Test PASSED] rho(A)={rho_A:.4f}, Gramian top={int(np.argmax(scores))}, "
        f"submod gains={[round(g, 2) for g in gains]}, MPC max_off={res_mpc['max_offtarget']:.3f} vs LQR={res_lqr['max_offtarget']:.3f}"
    )


if __name__ == "__main__":
    main()
