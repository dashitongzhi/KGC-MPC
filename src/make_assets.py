"""Generate IEEE-format serif figures and LaTeX tables from real Colab GPU CSV/JSON outputs for Paper 3 (KGC-MPC)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "manuscript" / "figures"
TAB_DIR = ROOT / "manuscript" / "tables"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.6,
        "figure.dpi": 300,
    }
)


def _box(ax, x: float, y: float, w: float, h: float, text: str, fc: str, ec: str, fs: float = 6.8, bold: bool = False) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w * 0.5,
        y + h * 0.5,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color="#1C2833",
    )


def _arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str = "#2C3E50", lw: float = 1.1) -> None:
    arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8.5, linewidth=lw, color=color)
    ax.add_patch(arr)


def make_fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.25))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Block 1: Koopman Lifting + Hill Saturation
    _box(
        ax, 0.015, 0.54, 0.21, 0.36,
        "Single-Cell State Lifting\n($x_0, x_{\\mathrm{pert}}, u$)\n"
        "$z_0 = W_{\\mathrm{lin}}^{\\top}\\tilde{x}_0 + \\alpha_{\\mathrm{res}}\\mathrm{MLP}(\\tilde{x}_0)$\n"
        "PCA Anchor $W_{\\mathrm{lin}}^{\\top} W_{\\mathrm{lin}} = I_r$",
        fc="#EBF5FB", ec="#1B4F72", fs=6.6, bold=True,
    )
    _box(
        ax, 0.015, 0.08, 0.21, 0.34,
        "Hill-Log Actuation Channel\n$h(u)_j = \\frac{u_j}{1 + \\alpha_j u_j}$, $\\alpha_j > 0$\n"
        "$B = W_{\\mathrm{act}}\\Phi_{\\mathrm{pert}} + B_\\Delta$\n(CRISPR or Log-Dose)",
        fc="#FEF9E7", ec="#B7950B", fs=6.6,
    )

    # Block 2: Schur-Stable Bilinear Dynamics
    _box(
        ax, 0.270, 0.22, 0.245, 0.68,
        "Schur-Stable Bilinear Koopman ID\n"
        "$A = U\\,\\mathrm{diag}(\\gamma_{\\mathrm{schur}}\\tanh\\sigma)\\,V^{\\top}$\n"
        "$\\Vert A \\Vert_2 \\leq \\gamma_{\\mathrm{schur}} = 0.95 < 1$\n\n"
        "$z_{t+1} = A z_t + B h(u_t) + \\sum_{\\ell=1}^{q}(g_\\ell^{\\top}h(u_t))(p_\\ell^{\\top}z_t)r_\\ell$\n\n"
        "Linear Decode: $\\hat{x}_T = \\bar{x} + C_{\\mathrm{dec}} z_T$\n"
        "(Bounds multi-step Kreiss transients)",
        fc="#E8F8F5", ec="#117A65", fs=6.5,
    )

    _arrow(ax, 0.225, 0.72, 0.270, 0.72, color="#1B4F72")
    _arrow(ax, 0.225, 0.25, 0.270, 0.36, color="#B7950B")

    # Block 3: Gramian Submodular Cocktail & Ranking
    _box(
        ax, 0.562, 0.54, 0.20, 0.36,
        "Controllability Gramian $W_c(S)$\n"
        "$\\sum_{t=0}^{H-1} A^t B_S B_S^{\\top} (A^{\\top})^t + \\epsilon I_r$\n"
        "Greedy $(1 - 1/e)$ Cocktail $\\log\\det W_c(S)$\n"
        "Zero-Shot Energy Ranking $E_{\\min}(j)$",
        fc="#F4ECF7", ec="#6C3483", fs=6.4,
    )

    # Block 4: Safety-Constrained DARE MPC Loop
    _box(
        ax, 0.562, 0.08, 0.20, 0.34,
        "Safety-Tube & DARE MPC\n"
        "$\\min_{u_{0:H-1}\\in[0,u_{\\max}]}$ LQR + $\\Delta_T^{\\top}P_{\\mathrm{DARE}}\\Delta_T$\n"
        "s.t. $\\Vert P_{\\perp}(z_t - z_0)\\Vert_2 \\leq \\delta_{\\mathrm{safe}}$",
        fc="#FDEDEC", ec="#922B21", fs=6.5,
    )

    _arrow(ax, 0.515, 0.72, 0.562, 0.72, color="#117A65")
    _arrow(ax, 0.515, 0.32, 0.562, 0.25, color="#922B21")

    # Block 5: Clinical Perturbation Output
    _box(
        ax, 0.805, 0.24, 0.18, 0.62,
        "Closed-Loop Perturbation\nTherapeutic Decisions\n\n"
        "1. Combination Regimens\n(Norman & Sci-Plex4)\n\n"
        "2. $10\\,\\mu\\mathrm{M}$ Extrapolation\n(Sci-Plex3: 762k cells)\n\n"
        "3. Safe Dose Schedule\n($0\\%$ off-target breach)",
        fc="#EBF5FB", ec="#1B4F72", fs=6.4, bold=True,
    )
    _arrow(ax, 0.762, 0.72, 0.805, 0.66, color="#6C3483")
    _arrow(ax, 0.762, 0.25, 0.805, 0.38, color="#922B21")

    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "p3_fig1_kgc_mpc_architecture.pdf")
    plt.close(fig)


def make_fig3_spectra_rollout() -> None:
    sp = json.loads((RESULTS / "koopman_spectra_norman.json").read_text())
    ro = pd.read_csv(RESULTS / "rollout_stability.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))

    # (a) Complex Eigenvalue Plane
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    axes[0].plot(np.cos(theta), np.sin(theta), color="#7F8C8D", ls="--", lw=1.0, label="Unit Circle ($|z|=1.0$)")
    axes[0].plot(0.95 * np.cos(theta), 0.95 * np.sin(theta), color="#117A65", ls=":", lw=1.1, label="Schur Bound ($\\gamma=0.95$)")
    axes[0].scatter(sp["edmd_eig_real"], sp["edmd_eig_imag"], color="#D35400", marker="x", s=26, alpha=0.75, label=f"EDMD ($\\rho={sp['rho_edmd']:.3f}$)")
    axes[0].scatter(sp["kgc_eig_real"], sp["kgc_eig_imag"], color="#1B4F72", marker="o", s=20, alpha=0.80, label=f"KGC-MPC ($\\rho={sp['rho_kgc']:.3f}$)")
    axes[0].axhline(0.0, color="#CBD5E1", lw=0.6)
    axes[0].axvline(0.0, color="#CBD5E1", lw=0.6)
    axes[0].set_aspect("equal", "box")
    axes[0].set_xlim(-1.12, 1.15)
    axes[0].set_ylim(-1.12, 1.48)
    axes[0].set_xlabel("$\\mathrm{Re}(\\lambda)$")
    axes[0].set_ylabel("$\\mathrm{Im}(\\lambda)$")
    axes[0].set_title("(a) Norman ($r=64$) Koopman Eigenvalues")
    axes[0].legend(loc="upper center", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=5.7)
    axes[0].grid(True, alpha=0.22, ls="--", lw=0.5)
    nd = sp.get("nonnormal_diagnostics", {})
    if nd:
        dk = nd.get("KGC_Stiefel", {})
        de = nd.get("Unconstrained_EDMD", {})
        dn = nd.get("KGC_Unconstrained_A", {})
        axes[0].text(
            0.0, -1.03,
            f"Stiefel $\\sigma_{{\\max}}={dk.get('sigma_max', 0):.3f}, \\kappa(V)={dk.get('eigvec_cond', 0):.1f}$ | "
            f"No-Stiefel $\\sigma_{{\\max}}={dn.get('sigma_max', 0):.3f}, \\kappa(V)={dn.get('eigvec_cond', 0):.1f}$",
            ha="center", va="bottom", fontsize=5.1, color="#1C2833",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", alpha=0.9),
        )

    # (b) 20-step Rollout Stability
    steps = ro["step"].to_numpy()
    axes[1].plot(steps, ro["edmd_norm"], color="#D35400", ls="--", marker="s", ms=3.5, lw=1.6, label="Unconstrained EDMD (Kreiss gain)")
    axes[1].plot(steps, ro["kgc_norm"], color="#1B4F72", ls="-", marker="o", ms=3.8, lw=1.8, label="KGC-MPC ($\\Vert A\\Vert_2 \\leq 0.95$)")
    axes[1].set_xlabel("Forecast Horizon Step $t$")
    axes[1].set_ylabel("Latent Trajectory Norm $\\Vert z_t\\Vert_2$")
    axes[1].set_title("(b) 20-Step Forced Rollout Stability")
    axes[1].set_xlim(0.5, 20.5)
    axes[1].set_ylim(0.0, 3.75)
    axes[1].grid(True, alpha=0.25, ls="--", lw=0.5)
    axes[1].legend(loc="center right", bbox_to_anchor=(0.98, 0.56), frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=6.4)
    ratio = float(ro["edmd_norm"].iloc[-1] / ro["kgc_norm"].iloc[-1])
    ed_cond = sp.get("nonnormal_diagnostics", {}).get("Unconstrained_EDMD", {}).get("eigvec_cond", 73.4)
    axes[1].annotate(
        f"EDMD: {ro['edmd_norm'].iloc[-1]:.3f} ({ratio:.2f}$\\times$ KGC {ro['kgc_norm'].iloc[-1]:.3f})\n"
        f"$\\rho={sp['rho_edmd']:.3f}$, $\\kappa_2(V)={ed_cond:.1f}$",
        xy=(14, float(ro["edmd_norm"].iloc[13])),
        xytext=(4.5, 2.30),
        fontsize=6.0,
        color="#D35400",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#D35400", lw=0.9),
    )

    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_DIR / "p3_fig3_spectra_rollout.pdf")
    plt.close(fig)


def make_fig4_mpc_cocktail() -> None:
    mp = pd.read_csv(RESULTS / "safety_mpc_pareto.csv")
    mm = pd.read_csv(RESULTS / "safety_mpc_mismatch.csv")
    ck = pd.read_csv(RESULTS / "cocktail_submodular.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))

    # (a) Safety MPC Pareto Frontier across delta_safe (Unified FISTA-AL Nominal + Cross-Seed Mismatch)
    mpc_sub = mp[mp["method"] == "KGC-MPC (Constrained)"].sort_values("delta_safe")
    lqr_row = mp[mp["method"] == "Unconstrained LQR"].iloc[0]
    tm_sub = mm[mm["method"] == "KGC Tube-MPC (Plant Mismatch)"].sort_values("delta_safe")
    cl_sub = mm[mm["method"] == "Closed-Loop Receding Safe-MPC (Plant Mismatch)"].sort_values("delta_safe")
    ol_sub = mm[mm["method"] == "Open-Loop Safe-MPC (Plant Mismatch)"].sort_values("delta_safe")
    lqr_mm = mm[mm["method"] == "Box-LQR Baseline (Plant Mismatch)"].iloc[0]

    ds = mpc_sub["delta_safe"].to_numpy()
    off = mpc_sub["max_offtarget"].to_numpy()
    err = mpc_sub["target_error"].to_numpy()

    axes[0].plot(off, err, color="#1B4F72", marker="o", lw=1.7, ms=4.2, label="Nominal FISTA-MPC (0% viol.)")
    if len(tm_sub) > 0:
        axes[0].plot(tm_sub["max_offtarget"], tm_sub["target_error"], color="#0E6251", marker="s", ls="-", lw=1.6, ms=3.8, label="Tube-MPC (RPI, Mismatch)")
    axes[0].plot(cl_sub["max_offtarget"], cl_sub["target_error"], color="#117A65", marker="D", ls="--", lw=1.4, ms=3.6, label="CL Receding MPC (Mismatch)")
    axes[0].plot(ol_sub["max_offtarget"], ol_sub["target_error"], color="#922B21", marker="^", ls=":", lw=1.3, ms=3.8, label="OL Safe-MPC (Mismatch)")
    axes[0].scatter([float(lqr_row["max_offtarget"])], [float(lqr_row["target_error"])], color="#D35400", marker="*", s=105, zorder=6, label="Nominal DARE-LQR")
    axes[0].scatter([float(lqr_mm["max_offtarget"])], [float(lqr_mm["target_error"])], color="#C0392B", marker="X", s=55, zorder=6, label="Box-LQR (Mismatch)")

    for dval, ox, ey in zip(ds, off, err):
        if dval in (0.05, 0.15):
            lbl = "$\\delta=0.05$ (0%)" if dval == 0.05 else "$\\delta\\geq 0.15$ (0%)"
            dx, dy = (0.005, 0.015) if dval == 0.05 else (0.008, -0.025)
            axes[0].annotate(lbl, xy=(ox, ey), xytext=(ox + dx, ey + dy), fontsize=5.3, color="#1B4F72")
    tm_25 = tm_sub[tm_sub["delta_safe"] == 0.25].iloc[0] if len(tm_sub[tm_sub["delta_safe"] == 0.25]) else tm_sub.iloc[-1]
    axes[0].annotate(
        f"Tube-MPC ($\\delta=0.25$)\n0% viol., err={tm_25['target_error']:.3f}",
        xy=(float(tm_25["max_offtarget"]), float(tm_25["target_error"])),
        xytext=(float(tm_25["max_offtarget"]) - 0.040, float(tm_25["target_error"]) - 0.048),
        ha="center",
        fontsize=5.1,
        color="#0E6251",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#0E6251", lw=0.75),
    )
    axes[0].set_xlabel("Max Orthogonal Off-Target Excursion $\\Vert P_\\perp \\Delta z_t\\Vert_2$")
    axes[0].set_ylabel("Relative Target Error")
    axes[0].set_xlim(0.015, 0.235)
    axes[0].set_ylim(0.62, 1.02)
    axes[0].set_title("(a) Safety-Constrained MPC: Nominal & Plant Mismatch")
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=4.8, ncol=1)
    axes[0].grid(True, alpha=0.25, ls="--", lw=0.5)

    # (b) Submodular log det W_c(S)
    kk = ck["k"].to_numpy()
    axes[1].plot(kk, ck["logdet_submodular"], color="#1B4F72", marker="o", lw=1.8, ms=4.5, label="Submodular Greedy ($\\log\\det W_c$)")
    axes[1].plot(kk, ck["logdet_topnorm"], color="#117A65", marker="^", ls="--", lw=1.4, ms=4.2, label="Top-$\\Vert B_{:,j}\\Vert_2$ Ranking")
    axes[1].plot(kk, ck["logdet_random"], color="#7F8C8D", marker="s", ls=":", lw=1.3, ms=3.8, label="Random Perturbation Subset")
    axes[1].set_xlabel("Perturbation Cocktail Size $k$")
    axes[1].set_ylabel("$\\log\\det W_c(S)$ (Controllability Volume)")
    axes[1].set_xticks(kk)
    axes[1].set_title("(b) $(1 - 1/e)$ Submodular Actuator Selection")
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=6.1)
    axes[1].grid(True, alpha=0.25, ls="--", lw=0.5)

    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_DIR / "p3_fig4_mpc_cocktail.pdf")
    plt.close(fig)


def make_fig5_gramian_prioritization() -> None:
    gn = pd.read_csv(RESULTS / "gramian_prioritization_norman.csv")
    gs = pd.read_csv(RESULTS / "sciplex3_drug_prioritization.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))

    # (a) Norman (23 unseen singles): Directional Steering vs Undirected Magnitude Potency AUPRC
    order_n = ["KGC_MinEnergy", "KGC_Gramian_Align", "Input_Norm", "CoExpr_Prior", "Random"]
    labels_n = ["KGC\nMinEnergy", "KGC\nAlign", "Input\nNorm", "CoExpr\nPrior", "Random"]
    p_dir = gn[gn["criterion"] == "Directional_Steering"].set_index("method")["auprc"]
    p_mag = gn[gn["criterion"] == "Magnitude_Potency"].set_index("method")["auprc"]
    x = np.arange(len(order_n))
    w = 0.36
    v_dir = [float(p_dir.loc[m]) for m in order_n]
    v_mag = [float(p_mag.loc[m]) for m in order_n]
    axes[0].bar(x - w / 2, v_dir, width=w, color="#1B4F72", label="Directional Steering AUPRC")
    axes[0].bar(x + w / 2, v_mag, width=w, color="#D35400", label="Undirected Magnitude AUPRC")
    axes[0].text(x[0] - w / 2, v_dir[0] + 0.02, f"{v_dir[0]:.3f}", ha="center", va="bottom", fontsize=5.6, color="#1B4F72", fontweight="bold")
    axes[0].text(x[2] + w / 2, v_mag[2] + 0.02, f"{v_mag[2]:.3f}", ha="center", va="bottom", fontsize=5.6, color="#D35400", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_n, fontsize=6.4)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("AUPRC (23 Unseen Singles)")
    axes[0].set_title("(a) Norman Zero-Shot Target Prioritization")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=6.0)
    axes[0].grid(True, axis="y", alpha=0.25, ls="--", lw=0.5)

    # (b) Sci-Plex3 (188 drugs across A549, MCF7, K562): AUROC Comparison
    cls = ["A549", "MCF7", "K562"]
    order_s = ["KGC_Gramian", "KGC_MinEnergy", "LowDose_Prolif_Rank", "CPA_Target_Align", "Random"]
    colors_s = {"KGC_Gramian": "#1B4F72", "KGC_MinEnergy": "#2E86C1", "LowDose_Prolif_Rank": "#117A65", "CPA_Target_Align": "#D35400", "Random": "#7F8C8D"}
    names_s = {"KGC_Gramian": "KGC Gramian", "KGC_MinEnergy": "KGC MinEnergy", "LowDose_Prolif_Rank": "Low-Dose Marker", "CPA_Target_Align": "CPA Align", "Random": "Random"}
    sub_s = gs.set_index(["cell_line", "method"])["auroc"]
    xs = np.arange(len(cls))
    ws = 0.15
    for idx_m, m in enumerate(order_s):
        vals = [float(sub_s.loc[(c, m)]) for c in cls]
        axes[1].bar(xs + (idx_m - 2.0) * ws, vals, width=ws, color=colors_s[m], label=names_s[m])
    axes[1].text(xs[1] - 2.0 * ws, float(sub_s.loc[("MCF7", "KGC_Gramian")]) + 0.02, f"{float(sub_s.loc[('MCF7', 'KGC_Gramian')]):.2f}", ha="center", va="bottom", fontsize=5.3, color="#1B4F72", fontweight="bold")
    axes[1].axhline(0.5, color="#94A3B8", ls="--", lw=0.8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(cls, fontsize=7.0)
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_ylabel("AUROC ($10\\,\\mu$M Antiproliferative Hits)")
    axes[1].set_title("(b) Sci-Plex3 Drug Screening (188 Drugs)")
    axes[1].legend(loc="upper center", ncol=3, frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=5.3, columnspacing=0.6)
    axes[1].grid(True, axis="y", alpha=0.25, ls="--", lw=0.5)

    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_DIR / "p3_fig5_gramian_prioritization.pdf")
    plt.close(fig)


def make_tables_1_3() -> None:
    df = pd.read_csv(RESULTS / "perturb_splits_summary.csv")
    mp = pd.read_csv(RESULTS / "safety_mpc_pareto.csv")
    ck = pd.read_csv(RESULTS / "cocktail_submodular.csv")
    gn = pd.read_csv(RESULTS / "gramian_prioritization_norman.csv")

    # Table 1: Norman (4 splits) & Adamson (unseen_single) across 6 methods (mean ± std across 3 seeds)
    gm = df.groupby(["cohort", "split", "method"])[["all_mse", "de20_pearson", "de20_dir_acc"]].mean()
    gs = df.groupby(["cohort", "split", "method"])[["de20_pearson"]].std(ddof=0)
    m_order = [
        ("KGC-MPC (Bilinear)", "KGC-MPC (Bilinear)"),
        ("KGC-Linear (No Bilin)", "KGC-Linear (No Bilin.)"),
        ("Unconstrained-EDMD", "Unconstrained EDMD"),
        ("GEARS-CoExpr", "GEARS-CoExpr"),
        ("CPA-Latent", "CPA-Latent"),
        ("Linear-Additive", "Linear-Additive"),
    ]
    cols = [
        ("NORMAN", "combo_seen2"),
        ("NORMAN", "combo_seen1"),
        ("NORMAN", "combo_seen0"),
        ("NORMAN", "unseen_single"),
        ("ADAMSON", "unseen_single"),
    ]
    best_r = {c: max(float(gm.loc[(*c, m), "de20_pearson"]) for m, _ in m_order) for c in cols}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Multi-Split Perturbation Response Prediction Across 3 Seeds (DE20 Pearson $r$ Mean $\\pm$ Std $\\uparrow$ / Sign Acc.$\\uparrow$). Best DE20 $r$ in Bold.}",
        "\\label{tab:genetic_pert}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.6pt}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Norman \\texttt{seen2} & Norman \\texttt{seen1} & Norman \\texttt{seen0} & Norman \\texttt{unseen} & Adamson \\texttt{unseen} \\\\",
        "\\midrule",
    ]
    for m_key, m_lbl in m_order:
        cells = []
        for c in cols:
            r_v = float(gm.loc[(*c, m_key), "de20_pearson"])
            s_v = float(gs.loc[(*c, m_key), "de20_pearson"])
            d_v = float(gm.loc[(*c, m_key), "de20_dir_acc"])
            r_fmt = f"{r_v:.3f}$\\pm${s_v:.2f}"
            r_s = f"\\textbf{{{r_fmt}}}" if abs(r_v - best_r[c]) < 5e-4 else r_fmt
            cells.append(f"{r_s} / {d_v:.3f}")
        lines.append(f"{m_lbl} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    (TAB_DIR / "p3_tab1_genetic_pert.tex").write_text("\n".join(lines) + "\n")

    # Table 3: Unified Safety-Constrained MPC Frontier (Nominal + Nonlinear Plant Mismatch) & Gramian Summary
    mpc_sub = mp[mp["method"] == "KGC-MPC (Constrained)"].set_index("delta_safe")
    lqr_row = mp[mp["method"] == "Unconstrained LQR"].iloc[0]
    mm = pd.read_csv(RESULTS / "safety_mpc_mismatch.csv").set_index(["method", "delta_safe"])
    p_dir = gn[gn["criterion"] == "Directional_Steering"].set_index("method")
    p_mag = gn[gn["criterion"] == "Magnitude_Potency"].set_index("method")

    t3 = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Norman Control Guarantees: Unified FISTA-AL Nominal \\& Bilinear Mismatch MPC ($H=8$; LQR Viol.\\ vs.\\ $\\delta=0.15$) \\& Target Ranking ($n=23$).}",
        "\\label{tab:mpc_gramian}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.5pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "MPC Setting & Tgt Err.$\\downarrow$ & Off-Tgt$\\downarrow$ & Viol.$\\downarrow$ & Energy$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for ds_v in [0.05, 0.10, 0.15]:
        row = mpc_sub.loc[ds_v]
        lbl = f"Nom. FISTA-MPC ($\\delta={ds_v:.2f}$)"
        t3.append(f"{lbl} & {row['target_error']:.3f} & \\textbf{{{row['max_offtarget']:.4f}}} & \\textbf{{{row['violation_rate']*100:.1f}\\%}} & {row['control_energy']:.3f} \\\\")
    t3.append(f"Nom. DARE-LQR & \\textbf{{{lqr_row['target_error']:.3f}}} & {lqr_row['max_offtarget']:.4f} & {float(mp[(mp['method']=='Unconstrained LQR') & (mp['delta_safe']==0.15)]['violation_rate'].iloc[0])*100:.1f}\\% & {lqr_row['control_energy']:.3f} \\\\")
    t3.append("\\midrule")
    for m_k, ds_v, mlbl, bold_off in [
        ("KGC Tube-MPC (Plant Mismatch)", 0.05, "Tube Mism. ($\\delta=.05$)", True),
        ("Closed-Loop Receding Safe-MPC (Plant Mismatch)", 0.05, "CL Mism. ($\\delta=.05$)", False),
        ("Open-Loop Safe-MPC (Plant Mismatch)", 0.05, "OL Mism. ($\\delta=.05$)", False),
        ("KGC Tube-MPC (Plant Mismatch)", 0.25, "Tube Mism. ($\\delta=.25$)", False),
        ("Closed-Loop Receding Safe-MPC (Plant Mismatch)", 0.25, "CL Mism. ($\\delta=.25$)", True),
        ("Box-LQR Baseline (Plant Mismatch)", 0.15, "Box-LQR Mism.", False),
    ]:
        rm = mm.loc[(m_k, ds_v)]
        te_s = f"\\textbf{{{rm['target_error']:.3f}}}" if (m_k.startswith("KGC Tube-MPC") and ds_v == 0.25) else f"{rm['target_error']:.3f}"
        off_s = f"\\textbf{{{rm['max_offtarget']:.4f}}}" if bold_off else f"{rm['max_offtarget']:.4f}"
        t3.append(f"{mlbl} & {te_s} & {off_s} & {rm['violation_rate']*100:.1f}\\% & {rm['control_energy']:.3f} \\\\")
    t3 += [
        "\\midrule",
        "Zero-Shot Ranking & Dir.ROC$\\uparrow$ & Dir.PRC$\\uparrow$ & Mag.ROC$\\uparrow$ & Mag.PRC$\\uparrow$ \\\\",
        "\\midrule",
    ]
    for mk, ml in [("KGC_MinEnergy", "KGC MinEnergy"), ("KGC_Gramian_Align", "KGC Gramian"), ("Input_Norm", "$\\|B_{:,j}\\|_2$ Norm"), ("CoExpr_Prior", "CoExpr Prior"), ("Random", "Random")]:
        d_au, d_ap = float(p_dir.loc[mk, "auroc"]), float(p_dir.loc[mk, "auprc"])
        m_au, m_ap = float(p_mag.loc[mk, "auroc"]), float(p_mag.loc[mk, "auprc"])
        s1 = f"\\textbf{{{d_au:.3f}}}" if mk == "CoExpr_Prior" else f"{d_au:.3f}"
        s2 = f"\\textbf{{{d_ap:.3f}}}" if mk == "KGC_MinEnergy" else f"{d_ap:.3f}"
        s3 = f"\\textbf{{{m_au:.3f}}}" if mk == "CoExpr_Prior" else f"{m_au:.3f}"
        s4 = f"\\textbf{{{m_ap:.3f}}}" if mk == "Input_Norm" else f"{m_ap:.3f}"
        t3.append(f"{ml} & {s1} & {s2} & {s3} & {s4} \\\\")
    t3 += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]
    (TAB_DIR / "p3_tab3_mpc_gramian.tex").write_text("\n".join(t3) + "\n")


def make_fig2_multicohort_generalization() -> None:
    df_p = pd.read_csv(RESULTS / "perturb_splits_summary.csv")
    df_s3 = pd.read_csv(RESULTS / "sciplex3_10uM_holdout.csv")
    df_s4 = pd.read_csv(RESULTS / "sciplex4_dual_drug.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))

    # (a) Genetic Perturb-seq DE20_r across Norman & Adamson
    g = df_p.groupby(["cohort", "split", "method"])["de20_pearson"].mean()
    splits = [
        ("NORMAN", "combo_seen2", "N-seen2"),
        ("NORMAN", "combo_seen1", "N-seen1"),
        ("NORMAN", "combo_seen0", "N-seen0"),
        ("NORMAN", "unseen_single", "N-uns."),
        ("ADAMSON", "unseen_single", "A-uns."),
    ]
    m_list = [
        ("KGC-MPC (Bilinear)", "KGC-MPC", "#1B4F72"),
        ("KGC-Linear (No Bilin)", "KGC-Lin.", "#2E86C1"),
        ("GEARS-CoExpr", "GEARS", "#117A65"),
        ("CPA-Latent", "CPA", "#D35400"),
        ("Unconstrained-EDMD", "EDMD", "#7F8C8D"),
    ]
    x = np.arange(len(splits))
    w = 0.15
    for idx_m, (mk, ml, col) in enumerate(m_list):
        vals = [max(0.0, float(g.loc[(coh, sp, mk)])) for coh, sp, _ in splits]
        axes[0].bar(x + (idx_m - 2.0) * w, vals, width=w, color=col, label=ml)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s[2] for s in splits], fontsize=6.3)
    axes[0].set_ylim(0.0, 1.15)
    axes[0].set_ylabel("Top-20 DE Pearson $r$ (clipped $\\geq 0$)")
    axes[0].set_title("(a) Norman & Adamson (160k cells)")
    axes[0].legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=5.3, columnspacing=0.5)
    axes[0].grid(True, axis="y", alpha=0.25, ls="--", lw=0.5)

    # (b) Sci-Plex3 (762,795 cells) 10 uM Holdout Whole-Transcriptome MSE (lower better)
    cls = ["A549", "MCF7", "K562"]
    s3_m = [
        ("KGC-MPC (Hill-Koopman)", "KGC-MPC", "#1B4F72"),
        ("CPA-Dose-AE", "CPA-Dose", "#D35400"),
        ("Unconstrained-EDMD", "EDMD", "#7F8C8D"),
        ("LogDose-Linear", "LogDose", "#8E44AD"),
    ]
    s3_sub = df_s3.set_index(["cell_line", "method"])
    x3 = np.arange(len(cls))
    w3 = 0.19
    for idx_m, (mk, ml, col) in enumerate(s3_m):
        vals = [float(s3_sub.loc[(c, mk), "all_mse"]) for c in cls]
        axes[1].bar(x3 + (idx_m - 1.5) * w3, vals, width=w3, color=col, label=ml)
        if mk == "KGC-MPC (Hill-Koopman)":
            for ci, v in enumerate(vals):
                axes[1].text(x3[ci] + (idx_m - 1.5) * w3, v + 0.0007, f"{v:.4f}"[1:], ha="center", va="bottom", fontsize=5.1, color="#1B4F72", fontweight="bold")
    axes[1].set_xticks(x3)
    axes[1].set_xticklabels(cls, fontsize=6.8)
    axes[1].set_ylim(0.0, 0.033)
    axes[1].set_ylabel("All-Gene Extrap. MSE ($\\downarrow$)")
    axes[1].set_title("(b) Sci-Plex3 $10\\,\\mu$M Holdout (762k cells)")
    axes[1].legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=5.3, columnspacing=0.5)
    axes[1].grid(True, axis="y", alpha=0.25, ls="--", lw=0.5)

    # (c) Sci-Plex3 DE20_r + Sci-Plex4 Dual-Drug DE20_r
    cats = ["SP3-A549", "SP3-MCF7", "SP3-K562", "SP4-A549"]
    xc = np.arange(len(cats))
    kgc_r = [
        float(s3_sub.loc[("A549", "KGC-MPC (Hill-Koopman)"), "de20_pearson"]),
        float(s3_sub.loc[("MCF7", "KGC-MPC (Hill-Koopman)"), "de20_pearson"]),
        float(s3_sub.loc[("K562", "KGC-MPC (Hill-Koopman)"), "de20_pearson"]),
        float(df_s4[(df_s4["cell_line"] == "A549") & (df_s4["method"] == "KGC-MPC (Small-Gain Bilinear)")]["de20_pearson"].iloc[0]),
    ]
    cpa_r = [
        float(s3_sub.loc[("A549", "CPA-Dose-AE"), "de20_pearson"]),
        float(s3_sub.loc[("MCF7", "CPA-Dose-AE"), "de20_pearson"]),
        float(s3_sub.loc[("K562", "CPA-Dose-AE"), "de20_pearson"]),
        float(df_s4[(df_s4["cell_line"] == "A549") & (df_s4["method"] == "CPA-Dose-AE")]["de20_pearson"].iloc[0]),
    ]
    edmd_r = [
        float(s3_sub.loc[("A549", "Unconstrained-EDMD"), "de20_pearson"]),
        float(s3_sub.loc[("MCF7", "Unconstrained-EDMD"), "de20_pearson"]),
        float(s3_sub.loc[("K562", "Unconstrained-EDMD"), "de20_pearson"]),
        float(df_s4[(df_s4["cell_line"] == "A549") & (df_s4["method"] == "Unconstrained-EDMD")]["de20_pearson"].iloc[0]),
    ]
    wc = 0.24
    axes[2].bar(xc - wc, kgc_r, width=wc, color="#1B4F72", label="KGC-MPC (Small-Gain / Hill)")
    axes[2].bar(xc, cpa_r, width=wc, color="#D35400", label="CPA-Dose-AE")
    axes[2].bar(xc + wc, edmd_r, width=wc, color="#7F8C8D", label="Unconstrained EDMD")
    for ci in (1, 2, 3):
        axes[2].text(xc[ci] - wc, kgc_r[ci] + 0.025, f"{kgc_r[ci]:.3f}", ha="center", va="bottom", fontsize=5.1, color="#1B4F72", fontweight="bold")
    sp4_unc_r = float(df_s4[(df_s4["cell_line"] == "A549") & (df_s4["method"] == "KGC-Bilinear (Unclamped)")]["de20_pearson"].iloc[0])
    axes[2].scatter([xc[3] - wc], [sp4_unc_r], color="#922B21", marker="v", s=34, zorder=5, label=f"Unclamped SP4 ({sp4_unc_r:.2f})")
    axes[2].set_xticks(xc)
    axes[2].set_xticklabels(cats, fontsize=6.1)
    axes[2].set_ylim(0.0, 1.34)
    axes[2].set_ylabel("Top-20 DE Pearson $r$ ($\\uparrow$)")
    axes[2].set_title("(c) High-Dose & Dual-Drug DE20 $r$")
    axes[2].legend(loc="upper center", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=4.8, columnspacing=0.4)
    axes[2].grid(True, axis="y", alpha=0.25, ls="--", lw=0.5)

    fig.tight_layout(pad=0.55)
    fig.savefig(FIG_DIR / "p3_fig2_multicohort_generalization.pdf")
    plt.close(fig)


def _fmt_sci(v: float) -> str:
    if abs(v) >= 100.0:
        exp = int(np.floor(np.log10(abs(v))))
        mant = v / (10.0 ** exp)
        return f"${mant:.2f}\\!\\times\\!10^{exp}$"
    if abs(v) >= 1.0:
        return f"{v:.2f}"
    return f"{v:.4f}"


def make_table_2() -> None:
    df_s3 = pd.read_csv(RESULTS / "sciplex3_10uM_holdout.csv").set_index(["cell_line", "method"])
    df_s4 = pd.read_csv(RESULTS / "sciplex4_dual_drug.csv").set_index(["cell_line", "method"])

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Chemical Perturbation Scaling on Sci-Plex3 ($N=762{,}795$ Cells, $188$ Drugs, $10\\,\\mu\\mathrm{M}$ Holdout) \\& Sci-Plex4 ($N=98{,}397$ Dual-Drug HDACi).}",
        "\\label{tab:sciplex_chemical}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.6pt}",
        "\\begin{tabular}{l|ccc|ccc|ccc}",
        "\\toprule",
        "\\multicolumn{1}{c|}{Sci-Plex3 ($10\\,\\mu\\mathrm{M}$ Holdout)} & \\multicolumn{3}{c|}{A549 ($244{,}281$ cells)} & \\multicolumn{3}{c|}{MCF7 ($344{,}862$ cells)} & \\multicolumn{3}{c}{K562 ($173{,}652$ cells)} \\\\",
        "Method & All MSE$\\downarrow$ & DE20 $r\\uparrow$ & Dir.$\\uparrow$ & All MSE$\\downarrow$ & DE20 $r\\uparrow$ & Dir.$\\uparrow$ & All MSE$\\downarrow$ & DE20 $r\\uparrow$ & Dir.$\\uparrow$ \\\\",
        "\\midrule",
    ]
    for mk, ml in [
        ("KGC-MPC (Hill-Koopman)", "KGC-MPC (Hill-Koopman)"),
        ("CPA-Dose-AE", "CPA-Dose-AE"),
        ("Unconstrained-EDMD", "Unconstrained EDMD"),
        ("LogDose-Linear", "LogDose-Linear"),
    ]:
        row_cells = []
        for cl in ["A549", "MCF7", "K562"]:
            r_row = df_s3.loc[(cl, mk)]
            mse_s = f"\\textbf{{{r_row['all_mse']:.4f}}}" if mk == "KGC-MPC (Hill-Koopman)" else f"{r_row['all_mse']:.4f}"
            is_best_r = (cl == "A549" and mk == "CPA-Dose-AE") or (cl in ("MCF7", "K562") and mk == "KGC-MPC (Hill-Koopman)")
            r_s = f"\\textbf{{{r_row['de20_pearson']:.3f}}}" if is_best_r else f"{r_row['de20_pearson']:.3f}"
            d_s = f"{r_row['de20_dir_acc']:.3f}"
            row_cells += [mse_s, r_s, d_s]
        lines.append(f"{ml} & " + " & ".join(row_cells) + " \\\\")

    lines += [
        "\\midrule",
        "\\multicolumn{1}{c|}{Sci-Plex4 Dual-Drug} & \\multicolumn{4}{c|}{A549 ($49{,}284$ cells, $15$ combos)} & \\multicolumn{5}{c}{MCF7 ($49{,}113$ cells, $15$ combos)} \\\\",
        "Method & All MSE$\\downarrow$ & DE20 MSE$\\downarrow$ & \\multicolumn{1}{c}{DE20 $r\\uparrow$} & \\multicolumn{1}{c|}{Dir.$\\uparrow$} & \\multicolumn{2}{c}{All MSE$\\downarrow$} & \\multicolumn{1}{c}{DE20 MSE$\\downarrow$} & DE20 $r\\uparrow$ & Dir.$\\uparrow$ \\\\",
        "\\midrule",
    ]
    s4_methods = [
        ("KGC-MPC (Small-Gain Bilinear)", "KGC-MPC (Small-Gain)"),
        ("KGC-Linear (No Bilin)", "KGC-Linear (No Bilin.)"),
        ("KGC-Bilinear (Unclamped)", "KGC-Bilinear (Unclamped)"),
        ("CPA-Dose-AE", "CPA-Dose-AE"),
        ("Unconstrained-EDMD", "Unconstrained EDMD"),
    ]
    best_a_mse = min(float(df_s4.loc[("A549", m), "all_mse"]) for m, _ in s4_methods)
    best_m_mse = min(float(df_s4.loc[("MCF7", m), "all_mse"]) for m, _ in s4_methods)
    best_a_r = max(float(df_s4.loc[("A549", m), "de20_pearson"]) for m, _ in s4_methods)
    best_m_r = max(float(df_s4.loc[("MCF7", m), "de20_pearson"]) for m, _ in s4_methods)
    for mk, ml in s4_methods:
        a = df_s4.loc[("A549", mk)]
        m = df_s4.loc[("MCF7", mk)]
        a_mse = f"\\textbf{{{a['all_mse']:.4f}}}" if abs(float(a["all_mse"]) - best_a_mse) < 1e-6 else _fmt_sci(float(a["all_mse"]))
        m_mse = f"\\textbf{{{m['all_mse']:.4f}}}" if abs(float(m["all_mse"]) - best_m_mse) < 1e-6 else _fmt_sci(float(m["all_mse"]))
        a_de = _fmt_sci(float(a["de20_mse"]))
        m_de = _fmt_sci(float(m["de20_mse"]))
        ar = f"\\textbf{{{a['de20_pearson']:.3f}}}" if abs(float(a["de20_pearson"]) - best_a_r) < 5e-4 else f"{a['de20_pearson']:.3f}"
        mr = f"\\textbf{{{m['de20_pearson']:.3f}}}" if abs(float(m["de20_pearson"]) - best_m_r) < 5e-4 else f"{m['de20_pearson']:.3f}"
        lines.append(
            f"{ml} & {a_mse} & {a_de} & \\multicolumn{{1}}{{c}}{{{ar}}} & \\multicolumn{{1}}{{c|}}{{{a['de20_dir_acc']:.3f}}} & "
            f"\\multicolumn{{2}}{{c}}{{{m_mse}}} & \\multicolumn{{1}}{{c}}{{{m_de}}} & {mr} & {m['de20_dir_acc']:.3f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    (TAB_DIR / "p3_tab2_sciplex_chemical.tex").write_text("\n".join(lines) + "\n")


def make_fig6_mpc_realtime_stress() -> None:
    rt = pd.read_csv(RESULTS / "mpc_realtime_scaling.csv")
    st = pd.read_csv(RESULTS / "mpc_stress_tests.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.45))

    # (a) Real-time solver latency (ms/step, log scale) across horizon H (r=64)
    hz = rt[rt["sweep_type"] == "horizon"].copy()
    solvers = [
        ("FISTA-AL (A100 B=64/traj)", "FISTA-AL (A100 B=64/traj)", "#2E86C1", "P", "-."),
        ("FISTA-AL (CPU BLAS Warm)", "FISTA-AL (CPU Warm)", "#1B4F72", "o", "-"),
        ("FISTA-AL (CPU BLAS Cold)", "FISTA-AL (CPU Cold)", "#5D6D7E", "v", ":"),
        ("CVXPY-SCS (Conic ADMM)", "CVXPY-SCS (Conic ADMM)", "#8E44AD", "D", "-."),
        ("CVXPY-Clarabel (SOCP)", "CVXPY-Clarabel (SOCP)", "#117A65", "^", "--"),
        ("Autograd-Adam (Orig. CPU)", "Autograd-Adam (CPU)", "#D35400", "s", ":"),
    ]
    for skey, slbl, col, mk, ls in solvers:
        sub = hz[hz["solver"] == skey].sort_values("horizon")
        if len(sub) > 0:
            axes[0, 0].plot(sub["horizon"], sub["mean_ms"], color=col, marker=mk, ls=ls, lw=1.45, ms=3.5, label=slbl)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks([4, 8, 12, 16, 24, 32])
    axes[0, 0].set_xlabel("Prediction Horizon $H$ ($r=64, k=8$)")
    axes[0, 0].set_ylabel("Solve Latency (ms/step, log)")
    axes[0, 0].set_title("(a) Horizon Scaling vs. Clarabel SOCP & SCS ADMM")
    axes[0, 0].legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=4.6, columnspacing=0.5)
    axes[0, 0].grid(True, which="both", alpha=0.22, ls="--", lw=0.5)
    cpu_32 = float(hz[(hz["solver"] == "FISTA-AL (CPU BLAS Warm)") & (hz["horizon"] == 32)]["mean_ms"].iloc[0])
    clar_32 = float(hz[(hz["solver"] == "CVXPY-Clarabel (SOCP)") & (hz["horizon"] == 32)]["mean_ms"].iloc[0])
    axes[0, 0].annotate(
        f"{clar_32/cpu_32:.1f}$\\times$ vs Clarabel\n({cpu_32:.1f} vs {clar_32:.0f} ms)",
        xy=(32, cpu_32),
        xytext=(17.0, 2.2),
        fontsize=5.0,
        color="#1B4F72",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1B4F72", lw=0.75),
    )

    # (b) A100 GPU batch-size crossover (B = 1..256) vs CPU BLAS (H=8, r=64)
    bs = rt[rt["sweep_type"] == "batch_size"].sort_values("batch_size")
    cpu_h8 = float(hz[(hz["solver"] == "FISTA-AL (CPU BLAS Warm)") & (hz["horizon"] == 8)]["mean_ms"].iloc[0])
    clar_h8 = float(hz[(hz["solver"] == "CVXPY-Clarabel (SOCP)") & (hz["horizon"] == 8)]["mean_ms"].iloc[0])
    axes[0, 1].plot(bs["batch_size"], bs["mean_ms"], color="#2E86C1", marker="o", lw=1.7, ms=4.0, label="FISTA-AL A100 (ms/traj)")
    axes[0, 1].axhline(cpu_h8, color="#1B4F72", ls="--", lw=1.3, label=f"CPU BLAS Warm ({cpu_h8:.2f} ms)")
    axes[0, 1].axhline(clar_h8, color="#117A65", ls=":", lw=1.3, label=f"Clarabel SOCP ({clar_h8:.1f} ms)")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(bs["batch_size"].tolist())
    axes[0, 1].set_xticklabels([str(int(b)) for b in bs["batch_size"].tolist()])
    axes[0, 1].set_xlabel("Parallel Trajectory Batch Size $B$ ($H=8, r=64$)")
    axes[0, 1].set_ylabel("Amortized Latency (ms/traj, log)")
    axes[0, 1].set_title("(b) A100 Batch Scaling & CPU-to-GPU Crossover ($B^\\star=8$)")
    axes[0, 1].legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=5.0)
    axes[0, 1].grid(True, which="both", alpha=0.22, ls="--", lw=0.5)
    b8_val = float(bs[bs["batch_size"] == 8]["mean_ms"].iloc[0])
    b256_val = float(bs[bs["batch_size"] == 256]["mean_ms"].iloc[0])
    axes[0, 1].annotate(
        f"Crossover $B^\\star=8$ ({b8_val:.2f} ms)\n$B=256 \\rightarrow {b256_val:.3f}$ ms/traj",
        xy=(8, b8_val),
        xytext=(18, 0.35),
        fontsize=5.0,
        color="#1B4F72",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1B4F72", lw=0.75),
    )

    # (c) Strong mismatch ($kappa in [1, 5]$, N=20): Satisfaction (%) & w/o RPI ablation
    mis = st[st["stress_axis"] == "mismatch_scale"].copy()
    m_styles_mis = [
        ("KGC Tube-MPC (RPI+Delay)", "KGC Tube-MPC (Full)", "#1B4F72", "o", "-"),
        ("Tube-MPC w/o RPI", "w/o RPI Tightening", "#2E86C1", "s", "--"),
        ("Std Receding Safe-MPC", "Std Receding MPC", "#117A65", "^", "-."),
        ("Box-LQR Baseline", "Box-LQR", "#D35400", "X", ":"),
    ]
    for mk, ml, col, mrk, ls in m_styles_mis:
        sub = mis[mis["method"] == mk].sort_values("stress_value")
        axes[1, 0].plot(sub["stress_value"], sub["constraint_sat_rate"] * 100.0, color=col, marker=mrk, ls=ls, lw=1.5, ms=3.8, label=ml)
    axes[1, 0].set_xticks([1.0, 2.0, 3.0, 5.0])
    axes[1, 0].set_ylim(45.0, 106.0)
    axes[1, 0].set_xlabel("Bilinear Plant Mismatch Scale $\\kappa_{\\mathrm{mis}}$ ($N=20$ targets)")
    axes[1, 0].set_ylabel("Tube Satisfaction Rate (%)")
    axes[1, 0].set_title("(c) Safety & RPI Ablation Under Mismatch ($N=20$)")
    axes[1, 0].legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=4.9)
    axes[1, 0].grid(True, alpha=0.25, ls="--", lw=0.5)
    lqr5 = mis[(mis["method"] == "Box-LQR Baseline") & (mis["stress_value"] == 5.0)].iloc[0]
    axes[1, 0].annotate(
        f"Box-LQR diverges\n(NormOff={lqr5['norm_offtarget_mean']:.1f} at $\\kappa=5$)",
        xy=(5.0, float(lqr5["constraint_sat_rate"]) * 100.0),
        xytext=(2.3, 62.0),
        fontsize=5.0,
        color="#D35400",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#D35400", lw=0.75),
    )

    # (d) Actuation delay stress (tau = 0..3, N=20): Normalized Drift + Component Ablations
    dly = st[st["stress_axis"] == "delay_steps"].copy()
    m_styles_dly = [
        ("KGC Tube-MPC (RPI+Delay)", "KGC Tube-MPC (Full)", "#1B4F72", "o", "-"),
        ("Tube-MPC w/o Delay-Pred", "w/o Delay-Predictor", "#8E44AD", "D", "--"),
        ("Tube-MPC w/o RPI", "w/o RPI Tightening", "#2E86C1", "s", "-."),
        ("Tube-MPC w/o Ancillary", "w/o Ancillary Feedback", "#16A085", "v", ":"),
        ("Std Receding Safe-MPC", "Std Receding MPC", "#117A65", "^", "-."),
        ("Box-LQR Baseline", "Box-LQR", "#D35400", "X", ":"),
    ]
    for mk, ml, col, mrk, ls in m_styles_dly:
        sub = dly[dly["method"] == mk].sort_values("stress_value")
        axes[1, 1].plot(sub["stress_value"], sub["norm_offtarget_mean"], color=col, marker=mrk, ls=ls, lw=1.45, ms=3.5, label=ml)
    axes[1, 1].axhline(1.0, color="#922B21", ls="--", lw=1.0, label="Tube Limit ($\\delta_{\\mathrm{safe}}$)")
    axes[1, 1].set_xticks([0, 1, 2, 3])
    axes[1, 1].set_ylim(0.25, 1.32)
    axes[1, 1].set_xlabel("Actuation Delay $\\tau$ (steps, $u_{\\max}=0.6$, $N=20$)")
    axes[1, 1].set_ylabel("Normalized Drift $\\Vert P_\\perp\\Delta z\\Vert_2 / \\delta_{\\mathrm{safe}}$")
    axes[1, 1].set_title("(d) Delay Stress & Component Ablation ($N=20$)")
    axes[1, 1].legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CBD5E1", fontsize=4.3, columnspacing=0.4)
    axes[1, 1].grid(True, alpha=0.25, ls="--", lw=0.5)

    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_DIR / "p3_fig6_mpc_realtime_stress.pdf")
    plt.close(fig)


def make_table_4_realtime_stress() -> None:
    rt = pd.read_csv(RESULTS / "mpc_realtime_scaling.csv")
    st = pd.read_csv(RESULTS / "mpc_stress_tests.csv")
    stats = pd.read_csv(RESULTS / "mpc_statistical_tests.csv").set_index(["regime", "comparator"])

    hz = rt[rt["sweep_type"] == "horizon"].set_index(["horizon", "solver"])
    dm = rt[rt["sweep_type"] == "dimension"].set_index(["latent_dim", "solver"])
    bs = rt[rt["sweep_type"] == "batch_size"].set_index(["batch_size", "solver"])
    stm = st.set_index(["stress_axis", "stress_value", "method"])

    def _sig_star(q_val: float) -> str:
        if q_val < 0.001:
            return "$^{**}$"
        if q_val < 0.05:
            return "$^{*}$"
        return ""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Real-Time FISTA-AL vs.\\ Clarabel/SCS \\& $N=20$ Target RPI Tube-MPC Ablations ($*$: BH-FDR $q<0.05$; $**$: $q<0.001$ vs.\\ Full).}",
        "\\label{tab:mpc_realtime_stress}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{0.9pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Solver ($k=8$) & Mean (ms) & p95 (ms) & Gap\\% & Sat.\\% \\\\",
        "\\midrule",
    ]
    for label, row in [
        ("FISTA A100 ($B{=}256$)", bs.loc[(256, "FISTA-AL (A100 Batch/traj)")]),
        ("FISTA A100 ($B{=}64,H{=}8$)", hz.loc[(8, "FISTA-AL (A100 B=64/traj)")]),
        ("FISTA CPU Warm ($H{=}8$)", hz.loc[(8, "FISTA-AL (CPU BLAS Warm)")]),
        ("FISTA CPU Cold ($H{=}8$)", hz.loc[(8, "FISTA-AL (CPU BLAS Cold)")]),
        ("SCS ADMM ($H{=}8$)", hz.loc[(8, "CVXPY-SCS (Conic ADMM)")]),
        ("Clarabel SOCP ($H{=}8$)", hz.loc[(8, "CVXPY-Clarabel (SOCP)")]),
        ("FISTA CPU ($H{=}32,r{=}64$)", hz.loc[(32, "FISTA-AL (CPU BLAS Warm)")]),
        ("SCS ADMM ($H{=}32$)", hz.loc[(32, "CVXPY-SCS (Conic ADMM)")]),
        ("Clarabel SOCP ($H{=}32$)", hz.loc[(32, "CVXPY-Clarabel (SOCP)")]),
        ("FISTA CPU ($H{=}8,r{=}256$)", dm.loc[(256, "FISTA-AL (CPU BLAS Warm)")]),
        ("Adam CPU ($H{=}8,r{=}64$)", hz.loc[(8, "Autograd-Adam (Orig. CPU)")]),
    ]:
        gap_s = "--" if pd.isna(row["opt_gap_pct"]) else f"{float(row['opt_gap_pct']):.2f}\\%"
        mean_v = float(row["mean_ms"])
        p95_v = float(row["p95_ms"])
        mean_s = f"{mean_v:.1f}" if mean_v >= 1000.0 else f"{mean_v:.2f}"
        p95_s = f"{p95_v:.1f}" if p95_v >= 1000.0 else f"{p95_v:.2f}"
        lines.append(
            f"{label} & {mean_s} & {p95_s} & {gap_s} & {float(row['constraint_sat_rate'])*100:.1f}\\% \\\\"
        )

    lines += [
        "\\midrule",
        "Regime & Method & Sat.\\%$\\uparrow$ & Norm.Off$\\downarrow$ & Tgt Err.$\\downarrow$ \\\\",
        "\\midrule",
    ]
    stress_rows = [
        ("Mismatch $\\kappa=5$", "Mismatch k=5", "mismatch_scale", 5.0, "KGC Tube-MPC (RPI+Delay)", "Tube (Full)", True, True, False),
        ("Mismatch $\\kappa=5$", "Mismatch k=5", "mismatch_scale", 5.0, "Tube-MPC w/o RPI", "w/o RPI", False, False, False),
        ("Mismatch $\\kappa=5$", "Mismatch k=5", "mismatch_scale", 5.0, "Std Receding Safe-MPC", "Std Receding", False, False, True),
        ("Mismatch $\\kappa=5$", "Mismatch k=5", "mismatch_scale", 5.0, "Box-LQR Baseline", "Box-LQR", False, False, False),
        ("Sat. $u_{\\max}=0.25$", "Sat u_max=0.25", "saturation_umax", 0.25, "KGC Tube-MPC (RPI+Delay)", "Tube (Full)", True, True, False),
        ("Sat. $u_{\\max}=0.25$", "Sat u_max=0.25", "saturation_umax", 0.25, "Std Receding Safe-MPC", "Std Receding", True, False, True),
        ("Delay $\\tau=3$", "Delay tau=3", "delay_steps", 3.0, "KGC Tube-MPC (RPI+Delay)", "Tube (Full)", True, False, False),
        ("Delay $\\tau=3$", "Delay tau=3", "delay_steps", 3.0, "Tube-MPC w/o Delay-Pred", "w/o Delay", True, False, False),
        ("Delay $\\tau=3$", "Delay tau=3", "delay_steps", 3.0, "Tube-MPC w/o Ancillary", "w/o Ancil.", True, True, False),
        ("Delay $\\tau=3$", "Delay tau=3", "delay_steps", 3.0, "Std Receding Safe-MPC", "Std Receding", True, False, True),
        ("Delay $\\tau=3$", "Delay tau=3", "delay_steps", 3.0, "Box-LQR Baseline", "Box-LQR", False, False, False),
    ]
    for reg_lbl, stat_reg, ax, val, mk, ml, b_sat, b_off, b_err in stress_rows:
        r = stm.loc[(ax, val, mk)]
        sat_s = f"\\textbf{{{float(r['constraint_sat_rate'])*100:.1f}\\%}}" if b_sat else f"{float(r['constraint_sat_rate'])*100:.1f}\\%"
        off_v = float(r["norm_offtarget_mean"])
        std_v = float(r["norm_offtarget_std"])
        off_raw = f"{off_v:.1f}" if off_v >= 10.0 else f"{off_v:.3f}$\\pm${std_v:.2f}"
        star = "" if mk == "KGC Tube-MPC (RPI+Delay)" else _sig_star(float(stats.loc[(stat_reg, mk), "q_norm_off"]))
        off_s = (f"\\textbf{{{off_raw}}}" if b_off else off_raw) + star
        err_v = float(r["target_error_mean"])
        err_raw = f"{err_v:.2f}" if err_v >= 10.0 else f"{err_v:.3f}"
        err_s = f"\\textbf{{{err_raw}}}" if b_err else err_raw
        lines.append(f"{reg_lbl} & {ml} & {sat_s} & {off_s} & {err_s} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]
    (TAB_DIR / "p3_tab4_mpc_realtime_stress.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    make_fig1_architecture()
    make_fig2_multicohort_generalization()
    make_fig3_spectra_rollout()
    make_fig4_mpc_cocktail()
    make_fig5_gramian_prioritization()
    make_fig6_mpc_realtime_stress()
    make_tables_1_3()
    make_table_2()
    make_table_4_realtime_stress()
    print("PAPER3_ALL_ASSETS_GENERATED_OK")
