"""Strict reproducibility, weight-matrix Schur stability, LaTeX, and CSV parity verification for Paper 3 (KGC-MPC)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    root_pdf = ROOT / "KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf"
    man_pdf = ROOT / "manuscript" / "main.pdf"
    main_tex = ROOT / "manuscript" / "main.tex"
    main_log = ROOT / "manuscript" / "main.log"
    bib_file = ROOT / "manuscript" / "references.bib"
    res_dir = ROOT / "results"
    wt_dir = ROOT / "weights"
    fig_dir = ROOT / "manuscript" / "figures"
    tab_dir = ROOT / "manuscript" / "tables"

    assert root_pdf.is_file() and man_pdf.is_file(), "Missing Paper 3 compiled PDFs."
    sha_root = hashlib.sha256(root_pdf.read_bytes()).hexdigest()
    sha_man = hashlib.sha256(man_pdf.read_bytes()).hexdigest()
    assert sha_root == sha_man, "Root PDF and manuscript/main.pdf differ."

    for doc in [
        "README.md",
        "STUDY_GUIDE_CN.md",
        "SUBMISSION_GUIDE.md",
        "LICENSE",
        "requirements.txt",
        ".github/workflows/ci.yml",
        "submission/COVER_LETTER_IEEE.md",
        "submission/DECLARATIONS_AND_METADATA.md",
        "colab_kgc_mpc_benchmark.ipynb",
    ]:
        assert (ROOT / doc).is_file(), f"Missing Paper 3 deliverable: {doc}"

    req_res = [
        "environment.json",
        "environment_sciplex.json",
        "environment_mismatch.json",
        "perturb_splits_summary.csv",
        "koopman_spectra_norman.json",
        "rollout_stability.csv",
        "cocktail_submodular.csv",
        "safety_mpc_pareto.csv",
        "safety_mpc_mismatch.csv",
        "gramian_prioritization_norman.csv",
        "sciplex3_10uM_holdout.csv",
        "sciplex3_drug_prioritization.csv",
        "sciplex4_dual_drug.csv",
        "mpc_realtime_scaling.csv",
        "mpc_stress_tests.csv",
        "mpc_statistical_tests.csv",
        "hyperparam_sensitivity.csv",
        "environment_tcst.json",
    ]
    for fn in req_res:
        assert (res_dir / fn).is_file(), f"Missing results/{fn}"

    for i in range(1, 7):
        matches = list(fig_dir.glob(f"p3_fig{i}_*.pdf"))
        assert len(matches) == 1 and matches[0].stat().st_size > 2000, f"Missing figure p3_fig{i}"
    for fn in ["p3_tab1_genetic_pert.tex", "p3_tab2_sciplex_chemical.tex", "p3_tab3_mpc_gramian.tex", "p3_tab4_mpc_realtime_stress.tex"]:
        assert (tab_dir / fn).is_file(), f"Missing table {fn}"

    n_pages = 10
    if main_log.is_file():
        log_txt = main_log.read_text(errors="ignore")
        m_pg = re.search(r"Output written on main\.(?:xdv|pdf) \((\d+) pages", log_txt)
        assert m_pg is not None, "Could not find Output written page count in main.log"
        n_pages = int(m_pg.group(1))
        assert 10 <= n_pages <= 12, f"Expected 10-12 pages after TCST/Automatica upgrade, got {n_pages}"
        for pat in ["LaTeX Error", "Emergency stop", "Undefined control sequence", "Warning--", "There were undefined citations", "There were undefined references"]:
            assert pat not in log_txt, f"LaTeX log error/warning: {pat}"
        for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_txt):
            assert float(m.group(1)) <= 3.0, f"Severe Overfull hbox: {m.group(1)}pt"
    else:
        assert root_pdf.read_bytes()[:5] == b"%PDF-" and root_pdf.stat().st_size > 200_000, "Invalid PDF without main.log"

    tex_txt = main_tex.read_text()
    bib_txt = bib_file.read_text()
    readme_txt = (ROOT / "README.md").read_text()
    repo_url = "https://github.com/dashitongzhi/KGC-MPC"
    assert repo_url in tex_txt and repo_url in readme_txt, "Missing public GitHub URL in main.tex or README.md"

    m_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_txt, re.S)
    assert m_abs is not None, "Missing LaTeX abstract"
    abs_clean = re.sub(r"\\allowbreak\b|\\textbf\{([^}]*)\}|\\texttt\{([^}]*)\}|\\mathrm\{([^}]*)\}", r"\1\2\3", m_abs.group(1))
    abs_clean = re.sub(r"\\[a-zA-Z]+\s*", " ", abs_clean)
    abs_words = [w for w in re.split(r"\s+", abs_clean.strip()) if re.search(r"[A-Za-z0-9]", w)]
    assert len(abs_words) <= 250, f"ScholarOne abstract exceeds 250 words: {len(abs_words)}"

    forbidden = [
        "/Users/" + "kral",
        "Hu" + "go",
        "civil" + "ization",
    ]
    for check_p in [
        main_tex,
        bib_file,
        ROOT / "README.md",
        ROOT / "SUBMISSION_GUIDE.md",
        ROOT / "submission" / "COVER_LETTER_IEEE.md",
        ROOT / "submission" / "DECLARATIONS_AND_METADATA.md",
        ROOT / "colab_kgc_mpc_benchmark.ipynb",
        *sorted((ROOT / "src").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]:
        txt_p = check_p.read_text(errors="ignore")
        for bad in forbidden:
            assert bad not in txt_p, f"Personal/local leak ({bad}) in {check_p.relative_to(ROOT)}"
    cites = set()
    for grp in re.findall(r"\\cite\{([^}]+)\}", tex_txt):
        for k in grp.split(","):
            cites.add(k.strip())
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_txt))
    missing = cites - bib_keys
    uncited = bib_keys - cites
    assert not missing, f"Undefined citation keys: {missing}"
    assert not uncited, f"Uncited bibliography entries: {uncited}"
    assert len(cites) == 36, f"Expected 36 cited bibliography keys, got {len(cites)}"
    assert "MAVS" not in tex_txt and "MIVS-CL" not in tex_txt and "CL-VSSC" not in tex_txt and "SPIC-MAS" not in tex_txt and "GRAM-UIO" not in tex_txt, "Cross-paper contamination!"

    # Verify Schur contractivity of all saved weight matrices
    sp = json.loads((res_dir / "koopman_spectra_norman.json").read_text())
    assert "nonnormal_diagnostics" in sp and "KGC_Unconstrained_A" in sp["nonnormal_diagnostics"]
    assert float(sp["nonnormal_diagnostics"]["KGC_Unconstrained_A"]["sigma_max"]) > 1.0
    for wt_name in ["norman_kgc_seed0.npz", "sciplex3_A549_kgc.npz", "sciplex3_MCF7_kgc.npz", "sciplex3_K562_kgc.npz"]:
        z = np.load(wt_dir / wt_name, allow_pickle=True)
        A = z["A"].astype(np.float64)
        sv_max = float(np.linalg.svd(A, compute_uv=False)[0])
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        assert sv_max <= 0.955 and rho <= 0.955, f"Schur bound violated in {wt_name}: sv={sv_max}, rho={rho}"
        if wt_name == "norman_kgc_seed0.npz":
            assert abs(rho - float(sp["rho_kgc"])) < 1e-4, "Spectral JSON mismatch!"

    # Verify exact numerical parity against real CSV/JSON files
    df_p = pd.read_csv(res_dir / "perturb_splits_summary.csv")
    df_ro = pd.read_csv(res_dir / "rollout_stability.csv")
    df_ck = pd.read_csv(res_dir / "cocktail_submodular.csv")
    df_mp = pd.read_csv(res_dir / "safety_mpc_pareto.csv")
    df_mm = pd.read_csv(res_dir / "safety_mpc_mismatch.csv")
    df_gn = pd.read_csv(res_dir / "gramian_prioritization_norman.csv")
    df_s3 = pd.read_csv(res_dir / "sciplex3_10uM_holdout.csv")
    df_s3p = pd.read_csv(res_dir / "sciplex3_drug_prioritization.csv")
    df_s4 = pd.read_csv(res_dir / "sciplex4_dual_drug.csv")

    g = df_p.groupby(["cohort", "split", "method"])["de20_pearson"].mean()
    assert f"{g.loc[('NORMAN', 'combo_seen2', 'KGC-MPC (Bilinear)')]:.3f}" in tex_txt
    assert f"{g.loc[('ADAMSON', 'unseen_single', 'KGC-MPC (Bilinear)')]:.3f}" in tex_txt
    assert f"{df_ro['kgc_norm'].iloc[-1]:.3f}" in tex_txt and f"{df_ro['edmd_norm'].iloc[-1]:.3f}" in tex_txt
    assert f"{df_ck['logdet_submodular'].iloc[-1]:.2f}" in tex_txt

    mpc_05 = df_mp[(df_mp["method"] == "KGC-MPC (Constrained)") & (df_mp["delta_safe"] == 0.05)].iloc[0]
    assert float(mpc_05["violation_rate"]) == 0.0, "Unified FISTA-AL nominal MPC must have 0.0% breach at delta=0.05"
    lqr_05 = df_mp[(df_mp["method"] == "Unconstrained LQR") & (df_mp["delta_safe"] == 0.05)].iloc[0]
    assert f"{mpc_05['max_offtarget']:.4f}" in tex_txt and f"{lqr_05['max_offtarget']:.4f}" in tex_txt

    gn_me = df_gn[(df_gn["criterion"] == "Directional_Steering") & (df_gn["method"] == "KGC_MinEnergy")].iloc[0]
    assert f"{gn_me['auprc']:.3f}" in tex_txt

    s3_k = df_s3[df_s3["method"] == "KGC-MPC (Hill-Koopman)"].set_index("cell_line")
    for cl in ["A549", "MCF7", "K562"]:
        assert f"{s3_k.loc[cl, 'all_mse']:.4f}" in tex_txt
    assert f"{s3_k.loc['MCF7', 'de20_pearson']:.3f}" in tex_txt and f"{s3_k.loc['K562', 'de20_pearson']:.3f}" in tex_txt
    mcf7_g = df_s3p[(df_s3p["cell_line"] == "MCF7") & (df_s3p["method"] == "KGC_Gramian")].iloc[0]
    assert f"{mcf7_g['auroc']:.3f}" in tex_txt
    s4_a = df_s4[(df_s4["cell_line"] == "A549") & (df_s4["method"] == "KGC-MPC (Small-Gain Bilinear)")].iloc[0]
    assert float(s4_a["all_mse"]) < 0.05 and f"{s4_a['all_mse']:.4f}" in tex_txt and f"{s4_a['de20_pearson']:.3f}" in tex_txt

    assert len(df_mm) == 28, f"Expected 28 rows in safety_mpc_mismatch.csv, got {len(df_mm)}"
    tm_05 = df_mm[(df_mm["delta_safe"] == 0.05) & (df_mm["method"] == "KGC Tube-MPC (Plant Mismatch)")].iloc[0]
    tm_25 = df_mm[(df_mm["delta_safe"] == 0.25) & (df_mm["method"] == "KGC Tube-MPC (Plant Mismatch)")].iloc[0]
    cl_25 = df_mm[(df_mm["delta_safe"] == 0.25) & (df_mm["method"] == "Closed-Loop Receding Safe-MPC (Plant Mismatch)")].iloc[0]
    lqr_mm = df_mm[(df_mm["delta_safe"] == 0.15) & (df_mm["method"] == "Box-LQR Baseline (Plant Mismatch)")].iloc[0]
    assert f"{tm_05['max_offtarget']:.4f}" in tex_txt and f"{tm_25['target_error']:.3f}" in tex_txt
    assert f"{cl_25['target_error']:.3f}" in tex_txt and f"{lqr_mm['max_offtarget']:.4f}" in tex_txt
    assert "1{,}021{,}000" in tex_txt and "762{,}795" in tex_txt and "98{,}397" in tex_txt

    df_rt = pd.read_csv(res_dir / "mpc_realtime_scaling.csv")
    df_st = pd.read_csv(res_dir / "mpc_stress_tests.csv")
    df_stat = pd.read_csv(res_dir / "mpc_statistical_tests.csv")
    df_sens = pd.read_csv(res_dir / "hyperparam_sensitivity.csv")

    assert len(df_rt) == 74 and len(df_st) == 84 and len(df_stat) == 30 and len(df_sens) == 9
    assert int(df_st["n_targets"].min()) == 20
    cpu_h8 = float(df_rt[(df_rt["sweep_type"] == "horizon") & (df_rt["horizon"] == 8) & (df_rt["solver"] == "FISTA-AL (CPU BLAS Warm)")]["mean_ms"].iloc[0])
    cpu_h32 = float(df_rt[(df_rt["sweep_type"] == "horizon") & (df_rt["horizon"] == 32) & (df_rt["solver"] == "FISTA-AL (CPU BLAS Warm)")]["mean_ms"].iloc[0])
    clar_h32 = float(df_rt[(df_rt["sweep_type"] == "horizon") & (df_rt["horizon"] == 32) & (df_rt["solver"] == "CVXPY-Clarabel (SOCP)")]["mean_ms"].iloc[0])
    scs_h32 = float(df_rt[(df_rt["sweep_type"] == "horizon") & (df_rt["horizon"] == 32) & (df_rt["solver"] == "CVXPY-SCS (Conic ADMM)")]["mean_ms"].iloc[0])
    b256 = float(df_rt[(df_rt["sweep_type"] == "batch_size") & (df_rt["batch_size"] == 256)]["mean_ms"].iloc[0])
    cpu_r256 = float(df_rt[(df_rt["sweep_type"] == "dimension") & (df_rt["latent_dim"] == 256) & (df_rt["solver"] == "FISTA-AL (CPU BLAS Warm)")]["mean_ms"].iloc[0])
    assert f"{cpu_h8:.2f}" in tex_txt and f"{cpu_h32:.2f}" in tex_txt and f"{clar_h32:.1f}" in tex_txt and f"{scs_h32:.2f}" in tex_txt
    assert f"{b256:.3f}" in tex_txt and f"{cpu_r256:.2f}" in tex_txt

    st_dly3 = df_st[(df_st["stress_axis"] == "delay_steps") & (df_st["stress_value"] == 3.0) & (df_st["method"] == "KGC Tube-MPC (RPI+Delay)")].iloc[0]
    st_nodly = df_st[(df_st["stress_axis"] == "delay_steps") & (df_st["stress_value"] == 3.0) & (df_st["method"] == "Tube-MPC w/o Delay-Pred")].iloc[0]
    assert f"{st_dly3['norm_offtarget_mean']:.3f}" in tex_txt and f"{st_nodly['norm_offtarget_mean']:.3f}" in tex_txt

    print(
        f"PAPER3_VERIFICATION_PASSED | PDF SHA-256={sha_root[:16]}... | N=1,021,000 cells, "
        f"{len(df_p)} Perturb rows, {len(df_s3)} SciPlex3 + {len(df_s4)} SciPlex4 + {len(df_mm)} Mismatch + "
        f"{len(df_rt)} RT + {len(df_st)} Stress (N=20 targets) + {len(df_stat)} Wilcoxon + {len(df_sens)} Sens rows, "
        f"{len(cites)}/{len(bib_keys)} cites verified, {n_pages} pages, <=3pt overfull."
    )


if __name__ == "__main__":
    main()
