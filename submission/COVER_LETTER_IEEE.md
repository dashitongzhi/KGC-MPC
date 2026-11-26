# Cover Letter Template — IEEE Transactions Submission

> **Target Journal**: *IEEE Transactions on Control Systems Technology*  
> *(Note: If submitting to Automatica or IEEE Transactions on Cybernetics instead, replace the journal name in the salutation, opening paragraph, and `manuscript/main.tex` `\markboth` line.)*  
> **Instructions**: Replace every bracketed `[TODO(PERSONAL_INFO): ...]` field below before converting to PDF or pasting into ScholarOne.

---

[TODO(PERSONAL_INFO): Date, e.g., November 27, 2026]

To the Editor-in-Chief and Editorial Board  
*IEEE Transactions on Control Systems Technology*

Dear Editor-in-Chief and Associate Editors,

We submit our manuscript entitled **"Schur-Stable Bilinear Koopman Control, Finite-Horizon Gramian Target Ranking, and Safety-Constrained MPC for Single-Cell Perturbation Steering"** for consideration as a Regular Paper in *IEEE Transactions on Control Systems Technology*.

Predicting transcriptional trajectories under combinatorial genetic perturbations and high-dose chemical screens while bounding off-target excursions poses a coupled system-identification and safety-critical optimal-control problem. Existing data-driven single-cell predictors operate in open loop without discrete-time singular-value contraction bounds, finite-horizon controllability guarantees, or closed-loop state constraints; meanwhile, unconstrained Dynamic Mode Decomposition (EDMD) suffers severe non-normal Kreiss transient norm inflation even when its spectral radius is small. Guided by modern multivariable control theory, we introduce **KGC-MPC**—a tightly integrated pipeline spanning Stiefel Schur system identification, submodular Gramian actuator design, condensed Augmented-Lagrangian FISTA optimization (`FISTA-AL`), and Robust Positive Invariant (RPI) delay-compensated Tube-MPC with formal Practical Input-to-State Stability (ISS) guarantees. This combination of control-theoretic rigor, solver real-time engineering, and large-scale empirical validation makes *IEEE Transactions on Control Systems Technology* an ideal venue for this work.

### Summary of Main Contributions
1. **Stiefel Polar-Factor Schur Contraction and Bilinear Small-Gain ISS (Proposition 1)**: Parameterizing the latent state matrix $A = U\,\mathrm{diag}(\gamma_{\mathrm{schur}}\tanh\sigma)\,V^\top$ via Stiefel polar retraction guarantees $\|A\|_2 \le 0.95 < 1$ and $\|A^t\|_2 \le 0.95^t$ for all $t \ge 1$. Paired with a zero-preserving residual observable, Hill-log receptor saturation, and small-gain bilinear coupling, KGC-MPC eliminates the $8.06\times$ 20-step Kreiss norm surge of unconstrained EDMD (`0.400` vs.\ `3.224`), whereas dropping Stiefel bounds destabilizes $\sigma_{\max}(A) = 1.0184$ and collapses unseen combination prediction.
2. **Finite-Horizon Controllability Gramian Actuator Design (Proposition 2)**: Establishes $(1 - 1/e)$ monotone submodular $\log\det W_c(S)$ cocktail selection ($-429.17 \to -379.75$) and zero-shot directional minimum-energy prioritization ($0.625$ AUPRC on 23 unseen Norman single perturbations).
3. **Condensed `FISTA-AL` Solver and Conic Benchmarks (Proposition 3)**: Eliminates state-trajectory decision variables in condensed Toeplitz coordinates, matching global `CVXPY-Clarabel SOCP` optimality within $0.17\%\text{--}0.30\%$ at $100.0\%$ constraint satisfaction while running in $4.45\text{--}18.61\,\mathrm{ms/step}$ on CPU BLAS ($110.9\times$ faster than `Clarabel SOCP` and $16.5\times$ faster than `SCS ADMM` at $H=32$). On an A100 GPU, batched screening crosses CPU latency at $B^\star=8$ ($5.24\,\mathrm{ms/traj}$) down to $0.168\,\mathrm{ms/traj}$ ($B=256$).
4. **Certified RPI Delay-Compensated Tube-MPC and 1,021,000-Cell Validation (Proposition 4, Algorithm 2, Theorem 1)**: Establishes recursive feasibility, RPI tube invariance, and Practical ISS under bounded mismatch and actuation delay. Across $N=20$ closed-loop targets and 12 severe stress regimes (240 episodes/controller spanning $5\times$ mismatch, $u_{\max}=0.25$ saturation, and $\tau=3$ delay), `KGC Tube-MPC` achieves $100.0\%$ recursive feasibility and $100.0\%$ tube satisfaction (paired Wilcoxon BH-FDR $q \le 1.80\times 10^{-39}$ vs.\ un-tightened MPC and $q_{\mathrm{sat}}=3.41\times 10^{-25}$ vs.\ Box-LQR). Across four real cohorts ($1{,}021{,}000$ single cells), KGC-MPC delivers the lowest $10\,\mu\mathrm{M}$ holdout MSE on all three Sci-Plex3 lines (`762,795` cells) and cuts Sci-Plex4 A549 dual-drug MSE $3.18\times$ to `0.0190`, while candidly reporting boundaries against `GEARS-CoExpr` on unexcited Norman singles and 1-step `CPA` on MCF7 dual screens.

### Originality, Reproducibility & Conflict of Interest
This manuscript is original, has not been published previously, and is not under concurrent review elsewhere. All source code, Colab GPU reproduction notebooks, trained model weights, figure/table generators, and verified experimental result archives are publicly available under the MIT license at `https://github.com/dashitongzhi/KGC-MPC`. The authors declare no competing financial or non-financial interests.

Thank you for your time and consideration.

Sincerely,

**[TODO(PERSONAL_INFO): Corresponding Author Full Name]**  
[TODO(PERSONAL_INFO): Title, Department / School, University / Institution]  
[TODO(PERSONAL_INFO): City, Postal Code, Country]  
E-mail: `[TODO(PERSONAL_INFO): author@institution.edu]`  
ORCID: `[TODO(PERSONAL_INFO): 0000-0000-0000-0000]`
