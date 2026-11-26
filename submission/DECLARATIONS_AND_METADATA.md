# ScholarOne / Editorial Manager Metadata & Declarations (Ready to Copy-Paste)

---

## 1. Manuscript Metadata

- **Full Title**:
  `Schur-Stable Bilinear Koopman Control, Finite-Horizon Gramian Target Ranking, and Safety-Constrained MPC for Single-Cell Perturbation Steering`
- **Short Running Title** (`<= 50` characters):
  `Schur-Stable Koopman Control & Safe Tube-MPC` *(44 chars)*
- **Article Type**: Regular Paper (or Original Research Article)
- **Target Primary Venue**: *IEEE Transactions on Control Systems Technology*
- **Alternative Venues**: *Automatica*, *IEEE Transactions on Cybernetics*, *IEEE Transactions on Control of Network Systems*
- **Manuscript Length**: 10 double-column `IEEEtran` pages (within standard 10–12 page IEEE Regular Paper budget; `0` Overfull hbox warnings)
- **GitHub Public Repository**: `https://github.com/dashitongzhi/KGC-MPC`

---

## 2. Plain-Text Abstract (IEEE ScholarOne `<= 250` Words Compliant)

> Copy-paste verbatim into the submission portal Abstract box (`215` words):

Predicting transcriptional responses to genetic perturbations and high-dose drugs while limiting off-target drift defines a coupled system-identification and optimal-control problem. Existing predictors (additive autoencoders, graph networks, and unconstrained dynamic mode decomposition, EDMD) lack singular-value contraction, finite-horizon controllability metrics, and closed-loop state constraints. We introduce KGC-MPC across 1,021,000 single cells (Norman, Adamson, Sci-Plex3, Sci-Plex4). First, Stiefel polar retraction (A = U diag(gamma_schur tanh(sigma)) V^T) guarantees ||A||_2 <= 0.95 < 1. Paired with a zero-preserving residual observable, Hill-log saturation, and small-gain bilinear coupling, KGC-MPC enforces Input-to-State Stability and prevents EDMD Kreiss inflation (0.400 vs. 3.224 20-step norm, 8.06x lower), whereas removing Stiefel bounds destabilizes sigma_max(A) = 1.0184. Second, finite-horizon controllability Gramians W_c(S) yield (1 - 1/e) submodular cocktails and zero-shot directional ranking (0.625 AUPRC). Third, condensed Augmented-Lagrangian FISTA MPC (FISTA-AL) with Robust Positive Invariant (RPI) tube tightening and delay compensation achieves 0.0% nominal violation across delta_safe in [0.05, 0.85] (-77.0% drift 0.0500 vs. 0.2177 LQR) and retains 100.0% recursive feasibility across N = 20 targets and 12 stress regimes (BH-FDR q <= 1.80e-39 vs. un-tightened MPC; q_sat = 3.41e-25 vs. Box-LQR). On CPU BLAS, FISTA-AL runs in 4.45-18.61 ms/step (110.9x faster than Clarabel SOCP, 16.5x over SCS ADMM at H = 32), while batched A100 evaluation reaches 0.168 ms/traj (B = 256). KGC-MPC leads 10 uM Sci-Plex3 MSE across all three lines (0.0188 A549, 0.0167 MCF7, 0.0123 K562) and cuts Sci-Plex4 A549 MSE 3.18x to 0.0190 (r = 0.950).

### Condensed Plain-Text Abstract for *Automatica* (`<= 200` Words Compliant, `154` words)

Predicting cellular transcriptional trajectories under genetic and chemical perturbations while restricting off-target drift defines a coupled system-identification and safety-constrained optimal-control problem. We present KGC-MPC across 1,021,000 single cells. First, Stiefel polar retraction enforces discrete-time singular-value contraction (||A||_2 <= 0.95 < 1), paired with zero-preserving observables, Hill-log dose saturation, and small-gain bilinear coupling to guarantee Input-to-State Stability and eliminate unconstrained EDMD Kreiss norm inflation (0.400 vs. 3.224 20-step norm). Second, finite-horizon controllability Gramians deliver (1 - 1/e) submodular actuator cocktails and directional minimum-energy target ranking. Third, condensed FISTA Augmented-Lagrangian MPC matches global conic SOCP accuracy (0.17%-0.30% gap) while running 110.9x faster than Clarabel SOCP and 16.5x faster than SCS ADMM on CPU BLAS, and scaling to 0.168 ms/traj on GPU (B = 256). Equipped with calibrated Robust Positive Invariant tube tightening, delay prediction, and ancillary DARE feedback, Tube-MPC certifies recursive feasibility and Practical ISS while holding 100.0% safety satisfaction across 240 severe mismatch, saturation, and delay stress episodes.

---

## 3. Keywords (`IEEE Index Terms`)

`Koopman operator, bilinear control, Schur stability, controllability Gramian, submodular optimization, safety-constrained MPC, Perturb-seq, Sci-Plex.`

---

## 4. Standard Compliance Declarations (Copy-Paste Ready)

- **Data and Code Availability Statement**:
  > All experiments analyze publicly available de-identified benchmark datasets: Norman et al. Perturb-seq CRISPRa (`GSE133344`), Adamson et al. Perturb-seq CRISPRi (`GSE90546`), and Srivatsan et al. Sci-Plex3 (`GSE139944`) and Sci-Plex4 (`GSM4150379`). Complete Python/PyTorch solver implementations, Clarabel/SCS scaling harnesses, Google Colab reproduction notebooks, trained model weights, figure/table scripts, and verified experimental archives are publicly available under the MIT license at `https://github.com/dashitongzhi/KGC-MPC`.
- **Declaration of Competing Interest**:
  > The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
- **Ethics & Consent Statement**:
  > This study performs secondary methodological analysis exclusively on publicly available, de-identified immortalized cancer cell-line (`K562`, `A549`, `MCF7`) perturbation datasets (`GSE133344`, `GSE90546`, `GSE139944`, `GSM4150379`). No new human participants, clinical trials, or animal experiments were conducted.
- **Funding Statement**:
  > `[TODO(PERSONAL_INFO): Insert grant agency and grant number, OR state "This research received no specific external grant from funding agencies in the public, commercial, or not-for-profit sectors."]`
- **CRediT Authorship Contribution Statement** *(for multi-author or Elsevier backup)*:
  > **[TODO(PERSONAL_INFO): Author 1 Name]**: Conceptualization, Methodology, Software, Formal analysis, Validation, Writing – original draft, Visualization. *(Add co-authors' Supervision / Funding acquisition / Writing – review & editing roles if applicable.)*

---

## 5. Elsevier / *Automatica* Backup Kit (`Highlights <= 85` Characters Each)

1. Schur-contractive bilinear Koopman ID cuts 20-step non-normal norm inflation 8.06x. *(82 chars)*
2. Submodular controllability Gramians guide (1-1/e) cocktails and target steering. *(80 chars)*
3. Condensed FISTA-AL solves in 4.45-18.61 ms/step, 110.9x faster than Clarabel SOCP. *(82 chars)*
4. Certified RPI delay-compensated Tube-MPC holds 100% feasibility over 240 stress runs. *(84 chars)*
5. Tested on 1,021,000 cells across 4 Perturb-seq and Sci-Plex dose-response screens. *(82 chars)*
