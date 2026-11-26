# Paper 3 — KGC-MPC: Schur-Stable Bilinear Koopman Control & Safety-Constrained MPC

[![CI Verification](https://github.com/dashitongzhi/KGC-MPC/actions/workflows/ci.yml/badge.svg)](https://github.com/dashitongzhi/KGC-MPC/actions/workflows/ci.yml)

- **Public GitHub Repository**: [`https://github.com/dashitongzhi/KGC-MPC`](https://github.com/dashitongzhi/KGC-MPC)
- **Compiled IEEE Manuscript (10 pages, `0 Overfull \hbox`)**: [KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf](KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf)
- **Turnkey Submission Suite & Journal Strategy**: [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md), [submission/COVER_LETTER_IEEE.md](submission/COVER_LETTER_IEEE.md), [submission/DECLARATIONS_AND_METADATA.md](submission/DECLARATIONS_AND_METADATA.md)
- **Chinese Defense & Formula Study Guide (SJTU-816)**: [STUDY_GUIDE_CN.md](STUDY_GUIDE_CN.md)
- **Colab GPU Reproduction Notebook (Auto-Syncs to Drive)**: [colab_kgc_mpc_benchmark.ipynb](colab_kgc_mpc_benchmark.ipynb)

---

## 1. Control-Theoretic Contribution

Reprogramming single-cell transcriptional trajectories under genetic (CRISPRa/CRISPRi) or chemical dose perturbations requires stabilizing learned Koopman dynamics, quantifying finite-horizon reachability, and bounding off-target drift:

$$z_{t+1} = A z_t + B h(u_t) + \Phi_\eta\!\Big(\sum_{\ell=1}^q (g_\ell^\top h(u_t))(p_\ell^\top z_t)\,r_\ell,\; z_t\Big), \qquad A = U\,\mathrm{diag}(\gamma_{\mathrm{schur}}\tanh\sigma)\,V^\top$$

1. **Stiefel Polar-Factor Schur Contraction & Bilinear Small Gain (Prop. 1)**: Enforces $\|A\|_2 \le 0.95 < 1$ and $\|A^t\|_2 \le 0.95^t$ for all $t \ge 1$ (eliminating non-normal Kreiss transient norm blow-up) together with Hill-log saturation $h(u)_j = u_j / (1 + \alpha_j u_j)$ and optional small-gain cone projection ($\eta = 0.04$).
2. **Finite-Horizon Controllability Gramian & $(1-1/e)$ Cocktails (Prop. 2)**: $W_c(S) = \epsilon I_r + \sum_{j \in S}\sum_{t=0}^{H-1}(A^t b_j)(A^t b_j)^\top$ provides monotone submodular $\log\det W_c(S)$ cocktail selection and directional minimum-energy $E_{\min}(j)$ unseen-target prioritization.
3. **Condensed FISTA-AL Safety-Tube MPC & Certified RPI Practical ISS (Props. 3–4, Algorithm 2, Theorem 1)**: Eliminates trajectory rollouts via condensed coordinates using a closed-form warm-started BLAS / batched CUDA `FISTA-AL` solver (`CondensedSafeMPCSolver`) enforcing $\|P_\perp(z_t - z_{\mathrm{ref}})\|_2 \le \delta_{\mathrm{safe}}$ with DARE terminal cost. Combining empirical cross-seed residual RPI tightening $r_t$, $\tau$-step state prediction, and ancillary DARE feedback establishes recursive feasibility, tube invariance, and Practical ISS (Theorem 1, proved via a 3-part Lyapunov argument in Appendix A).

---

## 2. Empirical Highlights (Real GPU Benchmarks, 1,021,000 Cells)

Executed across four real public perturbation cohorts (`Tesla T4` for Norman/Adamson; `NVIDIA L4` for Sci-Plex3/4; shared `NVIDIA A100-SXM4-80GB` for TCST conic scaling, batch crossover, $N=20$ target stress tests, Kreiss diagnostics, and hyperparameter sensitivity; `results/environment_tcst.json` notes `exclusive_gpu_verified: false`, `active_compute_procs_at_finish: 2`):

| Cohort / Benchmark | Real Scale | Key Result | Honest Comparison / Trade-Off |
|---|---|---|---|
| **Norman 2019** (GSE133344) | `91,205` CRISPRa cells (`3` seeds) | `combo_seen2` DE20 $r = \mathbf{0.948}$ (`99.9%` dir); `combo_seen0` `+0.046` over Linear (`0.260 vs 0.214`) | `GEARS-CoExpr` leads `unseen_single` (`0.477 vs 0.201`) via graph priors on unexcited $B_{:,j}$ |
| **Adamson 2016** (GSE90546) | `68,603` CRISPRi cells (`3` seeds) | `unseen_single` DE20 $r = \mathbf{0.728}$ (`90.3%` dir), **`+0.048`** over KGC-Linear (`0.680`) | Beats `EDMD` (`0.000`) & `CPA` (`-0.027`); behind `GEARS-CoExpr` (`0.796`) |
| **Non-Normal Kreiss & Sensitivity** | Norman ($r=64$) | `KGC_Stiefel` ($\rho=0.8888, \sigma_{\max}=0.8897, \kappa(V)=75.73, \mathcal{K}_{\mathrm{lb}}=1.00$) stabilizes `0.400`; $\gamma_{\mathrm{schur}}\in[0.80,0.99]$ holds `seen2` `0.938–0.941`, $q\in[0,24]$ holds `0.935–0.941` | Dropping Stiefel (`No-Stiefel`) raises $\sigma_{\max}=1.0184$, $\mathcal{K}_{\mathrm{lb}}=1.0035$, $\kappa(V)=106.30$, `ro20=0.463` and collapses `combo_seen0` to `-0.005`; `EDMD` inflates **$8.06\times$** (`3.224`) via Kreiss transient gain despite $\rho=0.0533$ |
| **Unified Safety MPC & Cocktail** | Norman ($H=8$, $k=1..8$) | Unified `FISTA-AL` hits **`0.0%` nominal viol.** across all $\delta_{\mathrm{safe}}\in[0.05,0.85]$ (**`-77.0%`** drift at $\delta=0.05$: `0.0500 vs 0.2177` LQR); mismatch `Tube-MPC` hits **`0.0%` viol.** everywhere (`0.0275 vs 0.1247` at $\delta=0.05$) and **#1 lowest target err** (`0.696`) at $\delta=0.25$ | `Box-LQR` violates `100.0%` for $\delta\le 0.15$; cocktail $\log\det W_c(S)$ rises `-429.17 -> -379.75` (`Top-Norm` `-381.09`); `KGC_MinEnergy` leads unseen directional AUPRC (`0.625`) |
| **Conic Scaling, Batch Crossover & $N=20$ Ablations** | $H=4..32$, $r=32..256$, $B=1..256$, $N=20$ targets $\times$ 12 regimes (`240` ep./ctrl) | CPU BLAS Warm **`4.45–18.61 ms/step`** ($H=4..32$: `6.33 ms` @ $H=8$, `18.61 ms` @ $H=32$; **`110.9×`** vs `CVXPY-Clarabel SOCP` `2063.1 ms`, **`16.5×`** vs `CVXPY-SCS ADMM` `307.74 ms`, **`255.9×`** vs `Autograd-Adam` `4762.4 ms`, `0.17%–0.30%` gap, `100.0%` sat.; `4.35->13.22 ms` across $r=32..256$). Batch A100 crosses CPU at **$B^\star=8$ (`5.24 ms/traj`)** down to **`0.168 ms/traj`** ($B=256$). `Tube-MPC (RPI+Delay)` hits **`100.0%` rec. feas. & `100.0%` sat.** across all 240 episodes (pooled `NormOff=0.425`) | Paired Wilcoxon BH-FDR ($N=20$): dropping RPI (`w/o RPI`) drops sat. to `90.0%` ($\kappa=1$) / `95.4%` ($\kappa=5$, `NormOff=0.703`, pooled $q=1.80\times 10^{-39}$); `w/o Delay-Pred` raises $\tau=3$ drift `+55.3%` (`0.531 vs 0.342`, $q=6.36\times 10^{-6}$); `w/o Ancillary` coasts with lower drift (`0.386`, $q=8.79\times 10^{-32}$) at higher tracking cost; `Box-LQR` fails (`70.2%` sat., $q_{\mathrm{sat}}=3.41\times 10^{-25}$; $\kappa=5$ `53.7` NormOff) |
| **Sci-Plex3** (GSE139944) | **`762,795`** cells (`188` drugs) | **#1 lowest $10\,\mu\mathrm{M}$ holdout MSE across all 3 lines**: `0.0188` (A549), `0.0167` (MCF7), `0.0123` (K562) | **#1 DE20 $r$** on MCF7 (`0.595`) & K562 (`0.523`); `MCF7` Gramian AUROC `0.701` (`Random 0.396`) |
| **Sci-Plex4** (GSM4150379) | `98,397` dual-drug HDACi cells | Zero-preserving MLP + `Small-Gain Bilinear` cuts A549 MSE **`3.18x`** (`0.0603 -> 0.0190`, **#1 lowest MSE**, `DE20 r = 0.950`) & MCF7 (`0.0525 -> 0.0352`) | Beats 1-step `EDMD` (`0.0199`) & `CPA` (`0.0255`) on A549 MSE; `CPA` (`0.0192`) & `EDMD` (`0.0230`) lead MCF7 dual screen |

---

## 3. Repository Structure & Reproducible Verification

```text
KGC-MPC/
├── KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf   # 10-page IEEE compiled manuscript (0 Overfull \hbox)
├── LICENSE / requirements.txt / .gitignore / .github/workflows/ci.yml
├── README.md / STUDY_GUIDE_CN.md / SUBMISSION_GUIDE.md / colab_kgc_mpc_benchmark.ipynb
├── submission/ (COVER_LETTER_IEEE.md, DECLARATIONS_AND_METADATA.md)
├── manuscript/ (main.tex, references.bib [36 keys], figures/p3_fig1..6, tables/p3_tab1..4)
├── src/        (kgc_mpc.py, baselines.py, data_loader.py, run_kgc_suite.py, run_tcst_benchmarks.py, make_assets.py, verify_paper.py)
├── tests/      (smoke_test.py)
├── results/    (18 real GPU CSV/JSON artifacts incl. mpc_statistical_tests.csv & hyperparam_sensitivity.csv)
└── weights/    (norman_kgc_seed0.npz, sciplex3_{A549,MCF7,K562}_kgc.npz)
```

### Verification Commands

```sh
pip install -r requirements.txt
python3 -B tests/smoke_test.py
python3 -B src/make_assets.py
cd manuscript && tectonic --keep-logs main.tex && cp main.pdf ../KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf && cd ..
python3 -B src/verify_paper.py
```

---

## 4. Target Venue & Submission Strategy (投稿建议)

- **首选主冲顶刊**：*IEEE Transactions on Control Systems Technology (IEEE TCST)*（完整 10 页标准 `IEEEtran` 双栏正文含附录证明，投稿前按 [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) 替换 3 处 `TODO(PERSONAL_INFO)` 即可直接提交 ScholarOne）。
- **理论导向备选**：*Automatica*（已备好 `<= 200` 词摘要与 5 条 `<= 85` 字符 `Highlights`）或 *IEEE Transactions on Cybernetics (IEEE TCYB)*。
- **独立性与伦理合规**：All code, theoretical propositions/theorems, experiments, and figures in this repository are strictly isolated and original to **KGC-MPC**, focusing on actuated Koopman system identification, controllability Gramians, and safety-constrained Tube-MPC across perturbation cohorts without cross-paper duplication.
