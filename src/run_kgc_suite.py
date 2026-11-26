"""Full Colab A100 experimental benchmark suite for Paper 3 (KGC-MPC)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _cand in reversed([str(_THIS_DIR), str(Path.cwd() / "src"), "/content"]):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from baselines import (  # noqa: E402
    UnconstrainedEDMD,
    compute_linear_additive_baseline,
    evaluate_perturbation_predictions,
    fit_and_predict_cpa,
    fit_and_predict_gears_gnn,
)
from data_loader import (  # noqa: E402
    load_gears_perturb_h5ad,
    load_sciplex4_dual_drug,
    load_sciplex_matrix_to_hvg,
    prepare_perturb_splits,
    SCIPLEX3_URLS,
)
from kgc_mpc import (  # noqa: E402
    compute_controllability_gramian,
    fit_kgc_system,
    get_device,
    score_targets_gramian,
    select_cocktail_submodular,
    solve_safety_mpc,
)


def top_de_indices(ctrl_mean: np.ndarray, target_mean: np.ndarray, k: int = 20) -> np.ndarray:
    return np.argsort(np.abs(target_mean - ctrl_mean))[::-1][:k]


def run_perturb_cohort(
    name: str,
    data_dir: Path,
    out_dir: Path,
    weights_dir: Path,
    seeds: List[int] = [0, 1, 2],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    print(f"\n=== [Paper 3] Cohort: {name.upper()} ===", flush=True)
    adata = load_gears_perturb_h5ad(name, data_dir)
    print(f"  Loaded {name}: shape={adata.shape}, conds={adata.obs.iloc[:, 0].nunique() if 'condition' not in adata.obs else adata.obs['condition'].nunique()}", flush=True)

    rows = []
    eig_payload: Dict[str, object] = {}
    rollout_rows = []
    prior_rows = []
    cocktail_rows = []
    mpc_rows = []

    for seed in seeds:
        prep = prepare_perturb_splits(adata, seed=seed, n_hvg=1000)
        X = prep["X"]
        U = prep["U"]
        conds = prep["conds"]
        ctrl_mask = prep["ctrl_mask"]
        pert_feats = prep["pert_feats"]
        split_map = prep["split_map"]

        x_ctrl = X[ctrl_mask]
        ctrl_mean = x_ctrl.mean(axis=0)
        train_conds = {c for c, sp in split_map.items() if sp == "train"}
        tr_mask = np.isin(conds, list(train_conds))
        x_tr, u_tr = X[tr_mask], U[tr_mask]

        # Build single-unit mean deltas from training singles only (for Linear-Additive baseline)
        train_unit_deltas: Dict[str, np.ndarray] = {}
        for c in train_conds:
            parts = [p for p in c.split("+") if p != "ctrl"]
            if len(parts) == 1:
                train_unit_deltas[parts[0]] = X[conds == c].mean(axis=0) - ctrl_mean

        # Fit KGC-System (Schur-stable Bilinear Koopman)
        t0 = time.perf_counter()
        kgc = fit_kgc_system(
            x_ctrl, x_tr, u_tr, prep["pert_list"],
            dose_mode=False, latent_dim=64, bilinear_rank=12, horizon=3, gamma_schur=0.95,
            epochs=55, batch_size=512, seed=seed, gene_pert_features=pert_feats,
        )
        t_kgc = time.perf_counter() - t0

        # Ablation: KGC-Linear (bilinear_rank=0)
        kgc_lin = fit_kgc_system(
            x_ctrl, x_tr, u_tr, prep["pert_list"],
            dose_mode=False, latent_dim=64, bilinear_rank=0, horizon=3, gamma_schur=0.95,
            epochs=55, batch_size=512, seed=seed, gene_pert_features=pert_feats,
        )

        # Unconstrained EDMDc
        edmd = UnconstrainedEDMD(latent_dim=64, reg=1e-3, horizon=3).fit(x_ctrl, x_tr, u_tr)

        if seed == 0 and name == "norman":
            A_k, B_k, C_k, _ = kgc.extract_matrices()
            eig_k = np.linalg.eigvals(A_k)
            eig_e = np.linalg.eigvals(edmd.A)
            eig_payload = {
                "kgc_eig_real": eig_k.real.tolist(),
                "kgc_eig_imag": eig_k.imag.tolist(),
                "edmd_eig_real": eig_e.real.tolist(),
                "edmd_eig_imag": eig_e.imag.tolist(),
                "rho_kgc": float(np.max(np.abs(eig_k))),
                "rho_edmd": float(np.max(np.abs(eig_e))),
            }
            np.savez_compressed(weights_dir / f"{name}_kgc_seed0.npz", A=A_k, B=B_k, C=C_k)

            # Multi-step unforced/forced rollout norm comparison H = 1..20
            dev = get_device()
            z0_k = kgc.encode(torch.tensor(ctrl_mean[None, :], dtype=torch.float32, device=dev))
            u0_k = torch.zeros((1, U.shape[1]), dtype=torch.float32, device=dev)
            u0_k[0, 0] = 1.0
            z0_e = np.zeros((1, edmd.A.shape[0]))
            _, traj_k = kgc.rollout(z0_k, u0_k, steps=20)
            traj_e = [z0_e.copy()]
            ze = z0_e.copy()
            u_e = np.zeros((1, U.shape[1])); u_e[0, 0] = 1.0
            for _ in range(20):
                ze = ze @ edmd.A.T + u_e @ edmd.B.T
                traj_e.append(ze.copy())
            for step in range(1, 21):
                rollout_rows.append({
                    "step": step,
                    "kgc_norm": float(torch.linalg.norm(traj_k[step]).cpu().item()),
                    "edmd_norm": float(np.linalg.norm(traj_e[step])),
                })

            # Gramian Target Prioritization & Submodular Cocktail & Safety-MPC
            # Build single-gene condition lookup (handles Norman/Adamson "GENE+ctrl" and "ctrl+GENE")
            single_cond_map: Dict[str, str] = {}
            for c_str in set(conds):
                if c_str == "ctrl":
                    continue
                parts = [p for p in str(c_str).split("+") if p != "ctrl" and p != ""]
                if len(parts) == 1:
                    single_cond_map[parts[0]] = str(c_str)

            true_potency = np.zeros(len(prep["pert_list"]))
            deltas_single = np.zeros((len(prep["pert_list"]), X.shape[1]), dtype=np.float32)
            single_seen = np.zeros(len(prep["pert_list"]), dtype=bool)
            for pi, gname in enumerate(prep["pert_list"]):
                sc_str = single_cond_map.get(gname)
                if sc_str is not None:
                    m_g = X[conds == sc_str].mean(axis=0) - ctrl_mean
                    deltas_single[pi] = m_g
                    true_potency[pi] = float(np.linalg.norm(m_g))
                    single_seen[pi] = True

            tr_single_idx = [
                pi for pi, g in enumerate(prep["pert_list"])
                if g in single_cond_map and split_map.get(single_cond_map[g], "") == "train"
            ]
            unseen_mask = single_seen & np.array([
                split_map.get(single_cond_map.get(g, ""), "") != "train" for g in prep["pert_list"]
            ])
            if len(tr_single_idx) >= 4 and np.sum(unseen_mask) >= 6:
                tr_pot = true_potency[tr_single_idx]
                tr_th = np.quantile(tr_pot, 0.80)
                tr_high = [pi for pi in tr_single_idx if true_potency[pi] >= tr_th]
                if not tr_high:
                    tr_high = [max(tr_single_idx, key=lambda idx: true_potency[idx])]
                d_tgt_vec = np.mean(deltas_single[tr_high], axis=0)
                d_tgt_unit = d_tgt_vec / (np.linalg.norm(d_tgt_vec) + 1e-8)
                dir_potency = deltas_single @ d_tgt_unit
                x_tgt = ctrl_mean + d_tgt_vec
                with torch.no_grad():
                    z_src = kgc.encode(torch.tensor(ctrl_mean[None, :], device=dev)).cpu().numpy()[0]
                    z_tgt = kgc.encode(torch.tensor(x_tgt[None, :], device=dev)).cpu().numpy()[0]

                scores_g, energies_g = score_targets_gramian(A_k, B_k, z_src, z_tgt, horizon=6)
                inv_energy_g = -np.log10(energies_g + 1e-6)
                tgt_coexpr = np.mean(pert_feats[tr_high], axis=0)
                coexpr_scores = pert_feats @ (tgt_coexpr / (np.linalg.norm(tgt_coexpr) + 1e-8))
                norm_scores = np.linalg.norm(B_k, axis=0)
                rng_r = np.random.default_rng(42)
                rand_scores = rng_r.normal(size=int(np.sum(unseen_mask)))
                for crit_name, val_arr in [
                    ("Directional_Steering", dir_potency[unseen_mask]),
                    ("Magnitude_Potency", true_potency[unseen_mask]),
                ]:
                    th_c = np.quantile(val_arr, 0.75)
                    yb = (val_arr >= th_c).astype(int)
                    if np.sum(yb) >= 2 and np.sum(1 - yb) >= 2:
                        for m_name, sc in [
                            ("KGC_Gramian_Align", scores_g[unseen_mask]),
                            ("KGC_MinEnergy", inv_energy_g[unseen_mask]),
                            ("Input_Norm", norm_scores[unseen_mask]),
                            ("CoExpr_Prior", coexpr_scores[unseen_mask]),
                            ("Random", rand_scores),
                        ]:
                            prior_rows.append({
                                "cohort": name.upper(),
                                "criterion": crit_name,
                                "n_unseen": int(np.sum(unseen_mask)),
                                "n_pos": int(yb.sum()),
                                "method": m_name,
                                "auroc": float(roc_auc_score(yb, sc)),
                                "auprc": float(average_precision_score(yb, sc)),
                            })
            else:
                z_src = np.zeros(A_k.shape[0], dtype=np.float32)
                z_tgt = np.ones(A_k.shape[0], dtype=np.float32) / np.sqrt(A_k.shape[0])

            # Submodular log det W_c(S) vs Random & Top-Norm greedy
            sel_sub, gains_sub = select_cocktail_submodular(A_k, B_k, budget_k=8, horizon=6)
            W_sub = np.eye(A_k.shape[0]) * 1e-3
            W_rand = np.eye(A_k.shape[0]) * 1e-3
            W_norm = np.eye(A_k.shape[0]) * 1e-3
            rand_perm = np.random.default_rng(42).permutation(B_k.shape[1])[:8]
            norm_order = np.argsort(np.linalg.norm(B_k, axis=0))[::-1][:8]
            for kk in range(1, 9):
                W_sub_k = compute_controllability_gramian(A_k, B_k[:, sel_sub[:kk]], horizon=6, reg_R=1e-3)
                W_rand_k = compute_controllability_gramian(A_k, B_k[:, rand_perm[:kk]], horizon=6, reg_R=1e-3)
                W_norm_k = compute_controllability_gramian(A_k, B_k[:, norm_order[:kk]], horizon=6, reg_R=1e-3)
                cocktail_rows.append({
                    "k": kk,
                    "logdet_submodular": float(np.linalg.slogdet(W_sub_k)[1]),
                    "logdet_topnorm": float(np.linalg.slogdet(W_norm_k)[1]),
                    "logdet_random": float(np.linalg.slogdet(W_rand_k)[1]),
                })

            # Safety-Constrained MPC vs Unconstrained LQR across delta_safe sweep
            dz_dir = (z_tgt - z_src) / (np.linalg.norm(z_tgt - z_src) + 1e-8)
            P_off = np.eye(A_k.shape[0]) - np.outer(dz_dir, dz_dir)
            # Restrict actuators to top-12 single targets for crisp clinical dose schedule
            B_top = B_k[:, sel_sub[:8]]
            lqr_res = solve_safety_mpc(A_k, B_top, z_src, z_tgt, P_off, delta_safe=0.40, u_max=1.0, horizon=8, enforce_constraints=False)
            for ds in [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.85]:
                mpc_res = solve_safety_mpc(A_k, B_top, z_src, z_tgt, P_off, delta_safe=ds, u_max=1.0, horizon=8, enforce_constraints=True)
                mpc_rows.append({
                    "delta_safe": ds,
                    "method": "KGC-MPC (Constrained)",
                    "target_error": mpc_res["relative_target_error"],
                    "max_offtarget": mpc_res["max_offtarget"],
                    "violation_rate": mpc_res["violation_rate"],
                    "control_energy": mpc_res["control_energy"],
                })
                mpc_rows.append({
                    "delta_safe": ds,
                    "method": "Unconstrained LQR",
                    "target_error": lqr_res["relative_target_error"],
                    "max_offtarget": lqr_res["max_offtarget"],
                    "violation_rate": float(np.mean(lqr_res["off_norms"] > ds)),
                    "control_energy": lqr_res["control_energy"],
                })

        # Evaluate across ALL split buckets with single CPA and GEARS training pass per seed (4x faster & cleaner GPU mem)
        split_buckets = sorted({sp for sp in split_map.values() if sp != "train"})
        bucket_data = []
        u_concat_list = []
        offset = 0
        for sp_name in split_buckets:
            sp_conds = sorted([c for c, s in split_map.items() if s == sp_name])
            if not sp_conds:
                continue
            u_test_list = []
            y_true_deltas = []
            de_idx_list = []
            combos_parsed = []
            for c in sp_conds:
                parts = [p for p in c.split("+") if p != "ctrl"]
                combos_parsed.append(tuple(parts))
                u_vec = np.zeros(U.shape[1], dtype=np.float32)
                for p in parts:
                    u_vec[prep["pert_to_idx"][p]] = 1.0
                u_test_list.append(u_vec)
                c_mean = X[conds == c].mean(axis=0)
                y_true_deltas.append(c_mean - ctrl_mean)
                de_idx_list.append(top_de_indices(ctrl_mean, c_mean, k=20))
            u_b = np.stack(u_test_list, axis=0)
            y_b = np.stack(y_true_deltas, axis=0)
            u_concat_list.append(u_b)
            bucket_data.append((sp_name, sp_conds, u_b, y_b, de_idx_list, combos_parsed, offset, offset + len(u_b)))
            offset += len(u_b)

        if bucket_data:
            u_all_te = np.concatenate(u_concat_list, axis=0)
            p_cpa_all = fit_and_predict_cpa(x_ctrl, x_tr, u_tr, u_all_te, dose_mode=False, epochs=35, seed=seed)
            p_gears_all = fit_and_predict_gears_gnn(x_ctrl, x_tr, u_tr, u_all_te, pert_feats, epochs=55, seed=seed)
            dev = get_device()
            for sp_name, sp_conds, u_test, y_true, de_idx_list, combos_parsed, s_i, e_i in bucket_data:
                with torch.no_grad():
                    x0_m = torch.tensor(ctrl_mean[None, :], dtype=torch.float32, device=dev).expand(len(u_test), -1)
                    ut_t = torch.tensor(u_test, dtype=torch.float32, device=dev)
                    p_kgc = kgc.forward_x(x0_m, ut_t)[0].cpu().numpy() - ctrl_mean[None, :]
                    p_kgc_lin = kgc_lin.forward_x(x0_m, ut_t)[0].cpu().numpy() - ctrl_mean[None, :]

                p_edmd = edmd.predict_delta(u_test)
                p_add = compute_linear_additive_baseline(ctrl_mean, train_unit_deltas, combos_parsed)
                preds_dict = {
                    "KGC-MPC (Bilinear)": p_kgc,
                    "KGC-Linear (No Bilin)": p_kgc_lin,
                    "Unconstrained-EDMD": p_edmd,
                    "GEARS-CoExpr": p_gears_all[s_i:e_i],
                    "CPA-Latent": p_cpa_all[s_i:e_i],
                    "Linear-Additive": p_add,
                }
                for m_name, y_p in preds_dict.items():
                    met = evaluate_perturbation_predictions(y_true, y_p, de_idx_list)
                    rows.append({
                        "cohort": name.upper(),
                        "split": sp_name,
                        "seed": seed,
                        "n_conds": len(sp_conds),
                        "method": m_name,
                        **met,
                    })
                    if seed == 0 and m_name == "KGC-MPC (Bilinear)":
                        print(f"  [{sp_name}] {m_name}: DE20_r={met['de20_pearson']:.3f}, DE20_MSE={met['de20_mse']:.4f}, dir={met['de20_dir_acc']:.3f}", flush=True)

        del kgc, kgc_lin, edmd, prep, X, U, x_tr, u_tr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del adata
    gc.collect()

    if rollout_rows:
        pd.DataFrame(rollout_rows).to_csv(out_dir / "rollout_stability.csv", index=False)
    if prior_rows:
        pd.DataFrame(prior_rows).to_csv(out_dir / "gramian_prioritization_norman.csv", index=False)
    if cocktail_rows:
        pd.DataFrame(cocktail_rows).to_csv(out_dir / "cocktail_submodular.csv", index=False)
    if mpc_rows:
        pd.DataFrame(mpc_rows).to_csv(out_dir / "safety_mpc_pareto.csv", index=False)

    return pd.DataFrame(rows), eig_payload


def run_sciplex3_suite(data_dir: Path, out_dir: Path, weights_dir: Path) -> None:
    print("\n=== [Paper 3] Cohort: Sci-Plex3 (GSE139944 ~649k cells, 188 drugs x 4 doses) ===", flush=True)
    X, meta, genes = load_sciplex_matrix_to_hvg(SCIPLEX3_URLS, data_dir, prefix="sciplex3", n_hvg=2000, min_umi=500)
    print(f"  Loaded Sci-Plex3: X={X.shape}, cell_types={meta['cell_type'].value_counts().to_dict()}", flush=True)

    # Identify proliferation arrest signature genes present in HVGs
    prolif_markers = [g for g in ["MKI67", "TOP2A", "CDK1", "PCNA", "UBE2C", "AURKA", "AURKB", "BIRC5", "CCNB1", "CCNB2", "MCM2"] if g in genes]
    prolif_idx = [genes.index(g) for g in prolif_markers]

    dose_rows = []
    drug_prior_rows = []

    for cl in ["A549", "MCF7", "K562"]:
        cl_mask = (meta["cell_type"].astype(str) == cl).to_numpy()
        X_cl = X[cl_mask]
        m_cl = meta.loc[cl_mask].reset_index(drop=True)

        veh_mask = (m_cl["vehicle"].astype(str).str.upper() == "TRUE").to_numpy() | (pd.to_numeric(m_cl["dose"], errors="coerce").fillna(0) == 0).to_numpy()
        doses = pd.to_numeric(m_cl["dose"], errors="coerce").fillna(0.0).to_numpy()
        drugs = m_cl["product_name"].fillna(m_cl["treatment"]).astype(str).to_numpy()

        # Valid non-vehicle drugs with >= 15 cells at low doses (10,100,1000) and >= 5 cells at 10000 nM
        uniq_drugs = sorted({d for d in drugs[~veh_mask] if d not in ("nan", "Vehicle", "DMSO", "None", "")})
        drug_to_idx = {d: i for i, d in enumerate(uniq_drugs)}

        x_ctrl = X_cl[veh_mask]
        ctrl_mean = x_ctrl.mean(axis=0)

        # Build input tensor U: log10(1 + dose_nM / 10.0) in column drug_idx
        non_veh = (~veh_mask) & np.isin(drugs, uniq_drugs)
        low_dose = non_veh & (doses <= 1000.0) & (doses > 0.0)
        high_dose = non_veh & (doses >= 9000.0)

        U_all = np.zeros((len(X_cl), len(uniq_drugs)), dtype=np.float32)
        nv_idx = np.where(non_veh)[0]
        for idx in nv_idx:
            U_all[idx, drug_to_idx[drugs[idx]]] = np.log10(1.0 + float(doses[idx]) / 10.0)

        x_tr, u_tr = X_cl[low_dose], U_all[low_dose]
        print(f"  [{cl}] ctrl={len(x_ctrl)}, train_low={len(x_tr)}, test_10uM={int(np.sum(high_dose))}, drugs={len(uniq_drugs)}", flush=True)

        # Fit KGC-System with Hill-log dose saturation (dose_mode=True)
        kgc = fit_kgc_system(
            x_ctrl, x_tr, u_tr, uniq_drugs,
            dose_mode=True, latent_dim=64, bilinear_rank=12, horizon=3, gamma_schur=0.95,
            epochs=45, batch_size=1024, seed=42,
        )
        A_k, B_k, C_k, alpha_k = kgc.extract_matrices()
        np.savez_compressed(weights_dir / f"sciplex3_{cl}_kgc.npz", A=A_k, B=B_k, C=C_k, alpha=alpha_k)

        edmd = UnconstrainedEDMD(latent_dim=64, reg=1e-2, horizon=3).fit(x_ctrl, x_tr, u_tr)

        # Held-out 10,000 nM trajectory prediction across all drugs present in high_dose
        test_drugs = []
        y_true_list = []
        de_list = []
        u_test_list = []
        dose_slopes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for d_name in uniq_drugs:
            d_low = low_dose & (drugs == d_name)
            d_hi = high_dose & (drugs == d_name)
            if np.sum(d_low) >= 8:
                # Fit simple per-drug linear log-dose slope from training doses
                dl_vals = np.log10(1.0 + doses[d_low] / 10.0)
                if len(np.unique(dl_vals)) >= 2:
                    X_d = X_cl[d_low] - ctrl_mean
                    slope = np.cov(dl_vals, X_d, rowvar=False)[0, 1:] / (np.var(dl_vals) + 1e-6)
                    intercept = X_d.mean(axis=0) - slope * np.mean(dl_vals)
                    dose_slopes[d_name] = (slope.astype(np.float32), intercept.astype(np.float32))
            if np.sum(d_hi) >= 6 and d_name in dose_slopes:
                test_drugs.append(d_name)
                m_hi = X_cl[d_hi].mean(axis=0)
                y_true_list.append(m_hi - ctrl_mean)
                de_list.append(top_de_indices(ctrl_mean, m_hi, k=20))
                uv = np.zeros(len(uniq_drugs), dtype=np.float32)
                uv[drug_to_idx[d_name]] = np.log10(1.0 + 10000.0 / 10.0)
                u_test_list.append(uv)

        u_test = np.stack(u_test_list, axis=0)
        y_true = np.stack(y_true_list, axis=0)
        dev = get_device()
        with torch.no_grad():
            x0_m = torch.tensor(ctrl_mean[None, :], dtype=torch.float32, device=dev).expand(len(u_test), -1)
            p_kgc = kgc.forward_x(x0_m, torch.tensor(u_test, dtype=torch.float32, device=dev))[0].cpu().numpy() - ctrl_mean[None, :]
        p_edmd = edmd.predict_delta(u_test)
        p_lin = compute_linear_additive_baseline(
            ctrl_mean, {}, [(d,) for d in test_drugs],
            test_doses=np.full(len(test_drugs), 10000.0), dose_unit_slopes=dose_slopes,
        )
        p_cpa = fit_and_predict_cpa(x_ctrl, x_tr, u_tr, u_test, dose_mode=True, epochs=30, batch_size=1024, seed=42)

        for m_name, yp in [
            ("KGC-MPC (Hill-Koopman)", p_kgc),
            ("CPA-Dose-AE", p_cpa),
            ("Unconstrained-EDMD", p_edmd),
            ("LogDose-Linear", p_lin),
        ]:
            met = evaluate_perturbation_predictions(y_true, yp, de_list)
            dose_rows.append({"cell_line": cl, "n_test_drugs": len(test_drugs), "method": m_name, **met})
            print(f"    [{cl} 10uM Holdout] {m_name}: DE20_r={met['de20_pearson']:.3f}, MSE={met['all_mse']:.4f}", flush=True)

        # Experiment 3B: Controllability Gramian Drug Prioritization for Antiproliferative Arrest
        if prolif_idx:
            # Ground truth arrest at 10 uM: proliferation signature drop relative to vehicle
            ctrl_prolif = float(x_ctrl[:, prolif_idx].mean())
            arrest_drop = []
            valid_drugs = []
            for d_name in uniq_drugs:
                d_hi = high_dose & (drugs == d_name)
                if np.sum(d_hi) >= 6:
                    drop = ctrl_prolif - float(X_cl[d_hi][:, prolif_idx].mean())
                    arrest_drop.append(drop)
                    valid_drugs.append(d_name)
            arrest_arr = np.array(arrest_drop)
            # Top 15% strongest antiproliferative compounds = positive ground-truth hits
            hit_thresh = np.quantile(arrest_arr, 0.85)
            y_hits = (arrest_arr >= hit_thresh).astype(int)
            # Target phenotype = mean state of known positive arrest signature (constructed by suppressing proliferation genes by 2 std)
            x_tgt = ctrl_mean.copy()
            for pi_g in prolif_idx:
                x_tgt[pi_g] = max(0.0, ctrl_mean[pi_g] - 1.5 * float(np.std(x_ctrl[:, pi_g])))
            with torch.no_grad():
                z_src = kgc.encode(torch.tensor(ctrl_mean[None, :], device=dev)).cpu().numpy()[0]
                z_tgt = kgc.encode(torch.tensor(x_tgt[None, :], device=dev)).cpu().numpy()[0]
            scores_g, energies_g = score_targets_gramian(A_k, B_k, z_src, z_tgt, horizon=6)
            inv_energy_g = -np.log10(energies_g + 1e-6)
            v_idx = [drug_to_idx[d] for d in valid_drugs]
            # LogFC baseline from low dose (1000 nM) proliferation drop
            low_drop = np.array([
                ctrl_prolif - float(X_cl[low_dose & (drugs == d)][:, prolif_idx].mean())
                if np.sum(low_dose & (drugs == d)) > 0 else 0.0
                for d in valid_drugs
            ])
            # CPA latent cosine alignment baseline
            cpa_align = np.array([
                float(np.dot(p_cpa[test_drugs.index(d)], x_tgt - ctrl_mean)) if d in test_drugs else 0.0
                for d in valid_drugs
            ])
            rng_r = np.random.default_rng(42)
            for m_name, sc in [
                ("KGC_Gramian", scores_g[v_idx]),
                ("KGC_MinEnergy", inv_energy_g[v_idx]),
                ("LowDose_Prolif_Rank", low_drop),
                ("CPA_Target_Align", cpa_align),
                ("Random", rng_r.normal(size=len(v_idx))),
            ]:
                top20 = np.argsort(sc)[::-1][:20]
                rec20 = float(np.sum(y_hits[top20]) / max(1, np.sum(y_hits)))
                drug_prior_rows.append({
                    "cell_line": cl,
                    "n_drugs": len(v_idx),
                    "n_hits": int(np.sum(y_hits)),
                    "method": m_name,
                    "auroc": float(roc_auc_score(y_hits, sc)),
                    "auprc": float(average_precision_score(y_hits, sc)),
                    "recall_at_20": rec20,
                })

        del kgc, edmd, X_cl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(dose_rows).to_csv(out_dir / "sciplex3_10uM_holdout.csv", index=False)
    pd.DataFrame(drug_prior_rows).to_csv(out_dir / "sciplex3_drug_prioritization.csv", index=False)
    del X, meta
    gc.collect()


def run_sciplex4_suite(data_dir: Path, out_dir: Path) -> None:
    print("\n=== [Paper 3] Cohort: Sci-Plex4 HDACi Dual-Drug Combinations ===", flush=True)
    X, meta, genes = load_sciplex4_dual_drug(data_dir, n_hvg=2000, min_umi=400)
    print(f"  Loaded Sci-Plex4: X={X.shape}, cell_lines={meta['cell_type'].value_counts().to_dict()}", flush=True)

    rows = []
    for cl in ["A549", "MCF7"]:
        cm = (meta["cell_type"] == cl).to_numpy()
        Xc = X[cm]
        mc = meta.loc[cm].reset_index(drop=True)
        veh = ((mc["dose1"] == 0) & (mc["dose2"] == 0)).to_numpy()
        if int(veh.sum()) < 20:
            continue
        x_ctrl = Xc[veh]
        ctrl_mean = x_ctrl.mean(axis=0)

        all_drugs = sorted((set(mc["drug1"]) | set(mc["drug2"])) - {"DMSO", "UNK", ""})
        d2i = {d: i for i, d in enumerate(all_drugs)}
        U = np.zeros((len(Xc), len(all_drugs)), dtype=np.float32)
        d1_arr = pd.to_numeric(mc["dose1"], errors="coerce").fillna(0.0).to_numpy()
        d2_arr = pd.to_numeric(mc["dose2"], errors="coerce").fillna(0.0).to_numpy()
        n1_arr = mc["drug1"].astype(str).to_numpy()
        n2_arr = mc["drug2"].astype(str).to_numpy()
        act1 = np.isin(n1_arr, all_drugs) & (d1_arr > 0.0)
        act2 = np.isin(n2_arr, all_drugs) & (d2_arr > 0.0)
        for idx in np.where(act1)[0]:
            U[idx, d2i[n1_arr[idx]]] = np.log10(1.0 + float(d1_arr[idx]))
        for idx in np.where(act2)[0]:
            U[idx, d2i[n2_arr[idx]]] = np.log10(1.0 + float(d2_arr[idx]))
        is_dual = act1 & act2

        # Hold out 35% of unique dual-drug conditions as unseen dual-drug combinations
        dual_conds = sorted(set(mc.loc[is_dual, "oligo"].apply(lambda s: "_".join(s.split("_")[:4]))))
        rng = np.random.default_rng(42)
        rng.shuffle(dual_conds)
        n_te = max(4, int(0.35 * len(dual_conds)))
        te_conds = set(dual_conds[:n_te])
        cond_key = mc["oligo"].apply(lambda s: "_".join(s.split("_")[:4])).to_numpy()
        te_mask = np.isin(cond_key, list(te_conds))
        tr_mask = (~veh) & (~te_mask) & (U.sum(axis=1) > 0)

        kgc_sg = fit_kgc_system(
            x_ctrl, Xc[tr_mask], U[tr_mask], all_drugs,
            dose_mode=True, latent_dim=48, bilinear_rank=10, horizon=3, gamma_schur=0.95,
            bilin_max_gain=0.04, epochs=45, batch_size=512, seed=42,
        )
        kgc_lin = fit_kgc_system(
            x_ctrl, Xc[tr_mask], U[tr_mask], all_drugs,
            dose_mode=True, latent_dim=48, bilinear_rank=0, horizon=3, gamma_schur=0.95,
            epochs=45, batch_size=512, seed=42,
        )
        kgc_unclamped = fit_kgc_system(
            x_ctrl, Xc[tr_mask], U[tr_mask], all_drugs,
            dose_mode=True, latent_dim=48, bilinear_rank=10, horizon=3, gamma_schur=0.95,
            bilin_max_gain=None, epochs=50, batch_size=256, seed=42,
        )
        edmd = UnconstrainedEDMD(latent_dim=48).fit(x_ctrl, Xc[tr_mask], U[tr_mask])

        u_te_list, y_te_list, de_list = [], [], []
        for ck in sorted(te_conds):
            m = cond_key == ck
            if m.sum() < 5:
                continue
            u_te_list.append(U[m][0])
            y_m = Xc[m].mean(axis=0)
            y_te_list.append(y_m - ctrl_mean)
            de_list.append(top_de_indices(ctrl_mean, y_m, k=20))

        u_te = np.stack(u_te_list, axis=0)
        y_te = np.stack(y_te_list, axis=0)
        dev = get_device()
        with torch.no_grad():
            x0_m = torch.tensor(ctrl_mean[None, :], dtype=torch.float32, device=dev).expand(len(u_te), -1)
            ut_t = torch.tensor(u_te, dtype=torch.float32, device=dev)
            p_kgc_sg = kgc_sg.forward_x(x0_m, ut_t)[0].cpu().numpy() - ctrl_mean[None, :]
            p_kgc_lin = kgc_lin.forward_x(x0_m, ut_t)[0].cpu().numpy() - ctrl_mean[None, :]
            p_kgc_unc = kgc_unclamped.forward_x(x0_m, ut_t)[0].cpu().numpy() - ctrl_mean[None, :]
        p_edmd = edmd.predict_delta(u_te)
        p_cpa = fit_and_predict_cpa(x_ctrl, Xc[tr_mask], U[tr_mask], u_te, dose_mode=True, epochs=35, seed=42)

        for m_name, yp in [
            ("KGC-MPC (Small-Gain Bilinear)", p_kgc_sg),
            ("KGC-Linear (No Bilin)", p_kgc_lin),
            ("KGC-Bilinear (Unclamped)", p_kgc_unc),
            ("CPA-Dose-AE", p_cpa),
            ("Unconstrained-EDMD", p_edmd),
        ]:
            met = evaluate_perturbation_predictions(y_te, yp, de_list)
            rows.append({"cell_line": cl, "n_test_dual": len(u_te), "method": m_name, **met})
            print(f"    [{cl} Dual-Drug] {m_name}: DE20_r={met['de20_pearson']:.3f}, DE20_MSE={met['de20_mse']:.4f}, all_MSE={met['all_mse']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(out_dir / "sciplex4_dual_drug.csv", index=False)


def run_norman_mpc_mismatch_check(data_dir: Path, out_dir: Path) -> None:
    """Evaluate open-loop and closed-loop receding-horizon Safe-MPC under real cross-seed nonlinear plant mismatch."""
    print("\n=== [Paper 3] Norman Plant-Model Mismatch Safe-MPC Verification ===", flush=True)
    adata = load_gears_perturb_h5ad("norman", data_dir)
    prep0 = prepare_perturb_splits(adata, seed=0, n_hvg=1000)
    prep1 = prepare_perturb_splits(adata, seed=1, n_hvg=1000)

    X0, U0, conds0, cm0, pf0, sm0 = prep0["X"], prep0["U"], prep0["conds"], prep0["ctrl_mask"], prep0["pert_feats"], prep0["split_map"]
    tr0 = np.isin(conds0, [c for c, sp in sm0.items() if sp == "train"])
    x_ctrl0 = X0[cm0]
    ctrl_mean0 = x_ctrl0.mean(axis=0)
    kgc_ctrl = fit_kgc_system(
        x_ctrl0, X0[tr0], U0[tr0], prep0["pert_list"],
        dose_mode=False, latent_dim=64, bilinear_rank=12, horizon=3, gamma_schur=0.95,
        epochs=55, batch_size=512, seed=0, gene_pert_features=pf0,
    )

    X1, U1, conds1, cm1, pf1, sm1 = prep1["X"], prep1["U"], prep1["conds"], prep1["ctrl_mask"], prep1["pert_feats"], prep1["split_map"]
    tr1 = np.isin(conds1, [c for c, sp in sm1.items() if sp == "train"])
    kgc_plant = fit_kgc_system(
        X1[cm1], X1[tr1], U1[tr1], prep1["pert_list"],
        dose_mode=False, latent_dim=64, bilinear_rank=12, horizon=3, gamma_schur=0.95,
        epochs=55, batch_size=512, seed=1, gene_pert_features=pf1,
    )

    A_k, B_k, _, _ = kgc_ctrl.extract_matrices()
    single_cond_map: Dict[str, str] = {}
    for c_str in set(conds0):
        if c_str != "ctrl":
            parts = [p for p in str(c_str).split("+") if p != "ctrl" and p != ""]
            if len(parts) == 1:
                single_cond_map[parts[0]] = str(c_str)
    tr_single = [pi for pi, g in enumerate(prep0["pert_list"]) if g in single_cond_map and sm0.get(single_cond_map[g], "") == "train"]
    deltas = {pi: X0[conds0 == single_cond_map[prep0["pert_list"][pi]]].mean(axis=0) - ctrl_mean0 for pi in tr_single}
    pot = {pi: float(np.linalg.norm(d)) for pi, d in deltas.items()}
    th = np.quantile(list(pot.values()), 0.80)
    high = [pi for pi in tr_single if pot[pi] >= th] or [max(tr_single, key=lambda idx: pot[idx])]
    d_tgt = np.mean([deltas[pi] for pi in high], axis=0)
    x_tgt = ctrl_mean0 + d_tgt

    dev = get_device()
    with torch.no_grad():
        z_src = kgc_ctrl.encode(torch.tensor(ctrl_mean0[None, :], device=dev)).cpu().numpy()[0]
        z_tgt = kgc_ctrl.encode(torch.tensor(x_tgt[None, :], device=dev)).cpu().numpy()[0]

    sel_sub, _ = select_cocktail_submodular(A_k, B_k, budget_k=8, horizon=6)
    B_top = B_k[:, sel_sub[:8]]
    dz_dir = (z_tgt - z_src) / (np.linalg.norm(z_tgt - z_src) + 1e-8)
    P_off = np.eye(A_k.shape[0]) - np.outer(dz_dir, dz_dir)

    p1_map = prep1["pert_to_idx"]
    ctrl_perts = [prep0["pert_list"][j] for j in sel_sub[:8]]
    plant_idx = [p1_map[g] for g in ctrl_perts]

    def _rollout_on_bilinear_plant(u_sched: np.ndarray, ctrl_ref: bool = False, delta_safe: float = 0.25) -> Dict[str, float]:
        x_cur = ctrl_mean0.copy()
        off_list = []
        energy = 0.0
        with torch.no_grad():
            z_p = kgc_plant.encode(torch.tensor(x_cur[None, :], dtype=torch.float32, device=dev))
            Ap, Bp = kgc_plant.get_A(), kgc_plant.get_B()
            for t in range(len(u_sched)):
                if ctrl_ref:
                    # Closed-loop receding horizon: measure current state in controller coordinates and re-plan!
                    z_obs = kgc_ctrl.encode(torch.tensor(x_cur[None, :], dtype=torch.float32, device=dev)).cpu().numpy()[0]
                    rem_h = max(1, len(u_sched) - t)
                    mpc_step = solve_safety_mpc(
                        A_k, B_top, z_obs, z_tgt, P_off,
                        delta_safe=delta_safe, u_max=1.0, horizon=rem_h, n_iters=140,
                        enforce_constraints=True, z_tube_ref=z_src,
                    )
                    ut_sub = mpc_step["u_traj"][0]
                else:
                    ut_sub = u_sched[t]
                energy += float(np.sum(ut_sub ** 2))
                u_full = torch.zeros((1, U1.shape[1]), dtype=torch.float32, device=dev)
                for k_loc, p_i in enumerate(plant_idx):
                    u_full[0, p_i] = float(ut_sub[k_loc])
                z_p = kgc_plant.step(z_p, u_full, A=Ap, B=Bp)
                x_cur = kgc_plant.decode(z_p).cpu().numpy()[0]
                z_ctrl_meas = kgc_ctrl.encode(torch.tensor(x_cur[None, :], dtype=torch.float32, device=dev)).cpu().numpy()[0]
                off_list.append(float(np.linalg.norm(P_off @ (z_ctrl_meas - z_src))))
        with torch.no_grad():
            z_fin = kgc_ctrl.encode(torch.tensor(x_cur[None, :], dtype=torch.float32, device=dev)).detach().cpu().numpy()[0]
        off_arr = np.array(off_list)
        return {
            "target_error": float(np.linalg.norm(z_fin - z_tgt) / (np.linalg.norm(z_src - z_tgt) + 1e-8)),
            "max_offtarget": float(np.max(off_arr)),
            "violation_rate": float(np.mean(off_arr > delta_safe)),
            "control_energy": energy,
        }

    # Note: Unconstrained LQR clipped to therapeutic box [0, u_max] for physical feasibility
    lqr_res = solve_safety_mpc(A_k, B_top, z_src, z_tgt, P_off, delta_safe=0.40, u_max=1.0, horizon=8, enforce_constraints=False)
    u_lqr_box = np.clip(lqr_res["u_traj"], 0.0, 1.0)
    mismatch_rows = []
    for ds in [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.85]:
        mpc_res = solve_safety_mpc(A_k, B_top, z_src, z_tgt, P_off, delta_safe=ds, u_max=1.0, horizon=8, enforce_constraints=True)
        ol_met = _rollout_on_bilinear_plant(mpc_res["u_traj"], ctrl_ref=False, delta_safe=ds)
        cl_met = _rollout_on_bilinear_plant(mpc_res["u_traj"], ctrl_ref=True, delta_safe=ds)
        lqr_met = _rollout_on_bilinear_plant(u_lqr_box, ctrl_ref=False, delta_safe=ds)
        for lbl, md in [
            ("Open-Loop Safe-MPC (Plant Mismatch)", ol_met),
            ("Closed-Loop Receding Safe-MPC (Plant Mismatch)", cl_met),
            ("Box-LQR Baseline (Plant Mismatch)", lqr_met),
        ]:
            mismatch_rows.append({"delta_safe": ds, "method": lbl, **md})
        print(
            f"  [Mismatch ds={ds:.2f}] CL-MPC max_off={cl_met['max_offtarget']:.4f} (viol={cl_met['violation_rate']:.2f}) "
            f"| OL-MPC={ol_met['max_offtarget']:.4f} | LQR={lqr_met['max_offtarget']:.4f}",
            flush=True,
        )

    pd.DataFrame(mismatch_rows).to_csv(out_dir / "safety_mpc_mismatch.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 3 KGC-MPC benchmark stages.")
    parser.add_argument("--stage", choices=["all", "perturb", "sciplex", "sciplex4", "mismatch", "review_fixes"], default="all")
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    t_start = time.perf_counter()
    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        work_dir = Path("/content") if Path("/content").exists() else Path(__file__).resolve().parents[1]
    data_dir = work_dir / "data"
    out_dir = work_dir / "results"
    weights_dir = work_dir / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "perturb"):
        df_norman, eig_norman = run_perturb_cohort("norman", data_dir, out_dir, weights_dir, seeds=[0, 1, 2])
        df_adamson, _ = run_perturb_cohort("adamson", data_dir, out_dir, weights_dir, seeds=[0, 1, 2])
        pd.concat([df_norman, df_adamson], ignore_index=True).to_csv(out_dir / "perturb_splits_summary.csv", index=False)
        (out_dir / "koopman_spectra_norman.json").write_text(json.dumps(eig_norman, indent=2))

    if args.stage in ("all", "sciplex"):
        run_sciplex3_suite(data_dir, out_dir, weights_dir)
        run_sciplex4_suite(data_dir, out_dir)
    elif args.stage == "sciplex4":
        run_sciplex4_suite(data_dir, out_dir)
    elif args.stage == "mismatch":
        run_norman_mpc_mismatch_check(data_dir, out_dir)
    elif args.stage == "review_fixes":
        run_sciplex4_suite(data_dir, out_dir)
        run_norman_mpc_mismatch_check(data_dir, out_dir)

    env_info = {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "stage": args.stage,
    }
    env_name = "environment_sciplex.json" if args.stage in ("sciplex", "sciplex4", "review_fixes") else ("environment_mismatch.json" if args.stage == "mismatch" else "environment.json")
    (out_dir / env_name).write_text(json.dumps(env_info, indent=2))
    print(f"\n[P3_SUITE_COMPLETE] Elapsed {env_info['elapsed_sec']}s on {env_info['gpu']}", flush=True)


if __name__ == "__main__":
    main()
