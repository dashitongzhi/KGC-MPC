# Paper 3（KGC-MPC）逐公式精读与上交 816 控制考研/面试答辩指南

> **公开代码仓**：[`https://github.com/dashitongzhi/KGC-MPC`](https://github.com/dashitongzhi/KGC-MPC)  
> **投稿操作指南**：[SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)（含 3 分钟 `TODO(PERSONAL_INFO)` 填空清单与 Cover Letter）  
> **对应论文**：[KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf](KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf)（10 页 IEEE 期刊双栏格式，`0 Overfull \hbox`）  
> **核心控制成果**：**Stiefel 极分解 Schur 收缩（Prop. 1）+ Hill–双线性小增益 ISS + 有限时域能控性格拉姆矩阵 $W_c(S)$（Prop. 2：$(1-1/e)$ 次模鸡尾酒与最小能量选靶）+ 凝聚态 FISTA-AL 实时优化（Prop. 3，胜 Clarabel SOCP $110.9\times$ / SCS ADMM $16.5\times$）+ 标定 RPI 递推可行与 Practical ISS 延迟补偿 Tube-MPC（Prop. 4, Algorithm 2, Theorem 1）**

---

## 1. 一句话电梯演讲（30 秒复试开场）

Paper 3 回答的是自动控制中的核心逆问题——**在多基因与多药物耦合的非线性生物调控网络中，如何学到一个长期前滚不发散的动力学被控对象，用最小控制能量选出协同药物/CRISPR 执行器组合，并在线毫秒级求解不越过脱靶安全管边界的最佳剂量序列？**  
传统基因扰动大模型（CPA、GEARS）只做开环静态单步预测，而标准动态模态分解（EDMD）即便谱半径 $\rho(A)=0.0533<1$，仍会因**非正规矩阵（Non-normal Kreiss Gain）瞬态放大**导致 $20$ 步滚动范数飙升 **$8.06\times$**（`3.224` vs `0.400`）。**KGC-MPC** 在 **1,021,000 个真实单细胞**（Norman + Adamson + Sci-Plex3 + Sci-Plex4）与 **$N=20$ 靶点 240 幕压力闭环实验**上实现：
1. 通过可微 SVD 极因子把 $A$ 的最大奇异值严格锁在 $\|A\|_2 \le 0.95 < 1$（Prop. 1），配零点保持提升 $\mathrm{MLP}(\tilde{x})-\mathrm{MLP}(\mathbf{0})$、Hill 对数剂量饱和与小增益双线性锥约束（在 `Sci-Plex4 A549` 上将 MSE 砍降 **$3.18\times$** 至 **`0.0190`** 全场第一，`DE20 r = 0.950`；`Sci-Plex3` 76 万细胞 $10\,\mu\mathrm{M}$ 外推全线 MSE 第一）；
2. 建立能控性格拉姆矩阵 $W_c(S)$ 的 $(1-1/e)$ 次模组合筛选与方向最小能量选靶（Prop. 2，23 个未知单基因方向 AUPRC **`0.625`** 排第一）；
3. 消除步进 rollout，提出凝聚态热启动 BLAS / 批处理 A100 **FISTA-AL**（CPU BLAS $4.45\text{--}18.61\text{ ms/step}$，比工业界 `CVXPY-Clarabel SOCP` 快 **`110.9\times`**、比 `CVXPY-SCS ADMM` 快 **`16.5\times`**，最优性间隙仅 `0.17%–0.30%`，在 $B^\star=8$ 交叉并达到 **`0.168 ms/traj`** @ $B=256$），并将名义安全 MPC 全面统一为 **`0.0%` 零越界**（$\delta_{\mathrm{safe}}=0.05$ 时脱靶漂移较 LQR 压低 **`77.0%`**）；
4. 提出带 RPI 管收紧、$\tau$ 步延迟预测与辅助 DARE 反馈的 **Tube-MPC（Algorithm 2, Theorem 1）**，在失配 $\kappa\le 5$、强饱和 $u_{\max}=0.25$、延迟 $\tau\le 3$ 共 12 组工况（240 幕/$N=20$ 靶点）全部保持 **`100.0%` 递推可行性与 `100.0%` 安全管满足率**，并通过**配对 Wilcoxon + BH-FDR 检验**统计显著战胜去 RPI（$q=1.80\times 10^{-39}$）、去延迟预测（$\tau=3$ $q=6.36\times 10^{-6}$）和 `Box-LQR`（$q_{\mathrm{sat}}=3.41\times 10^{-25}$）。

---

## 2. 现代控制理论与生信/药效调控的一一映射

| 上交 816 现代控制核心考点 | 论文数学表达与代码位置 |
|---|---|
| Koopman 非线性升维观察量（Eq. 1） | `W_lin` 正交 PCA 锚定 + 零点保持残差 $\mathrm{MLP}(\tilde{x})-\mathrm{MLP}(\mathbf{0})$ 线性解码 $\hat{x} = \bar{x} + C_{\mathrm{dec}} z$（`kgc_mpc.py:encode/decode`） |
| 严格 Schur 与 Kreiss 瞬态收缩（Eq. 2, Prop. 1i） | $A = U\,\mathrm{diag}(\gamma_{\mathrm{schur}}\tanh\sigma)\,V^\top$，$U,V$ 由可微 SVD 极投影产生 $\implies \|A^t\|_2 \le 0.95^t$ 与 Kreiss 常数 $\mathcal{K}(A)=1$，比仅约束特征值 $\rho(A)$ 更强！ |
| 执行器非线性饱和（Hill-Log Saturation） | $h(u)_j = \frac{u_j}{1 + \alpha_j u_j}$（$\alpha_j = \mathrm{softplus}(\alpha_{\mathrm{raw},j})$），刻画受体占有率对高剂量 $10\,\mu\mathrm{M}$ 外推的饱和物理律 |
| 双线性上位交互与小增益 ISS（Eq. 3, Prop. 1ii） | $z_{t+1} = A z_t + B h(u_t) + \Phi_\eta(\sum_{\ell=1}^q (g_\ell^\top h(u_t))(p_\ell^\top z_t)r_\ell)$；锥投影 $\|\Delta\|_2 \le \eta\|z\|_2$ 保证 $\gamma_{\mathrm{schur}}+\eta = 0.99 < 1$ |
| 有限时域能控性 Gramian（Eq. 4, Prop. 2） | $W_c(S) = \epsilon I_r + \sum_{j\in S}\sum_{t=0}^{H-1}(A^t b_j)(A^t b_j)^\top$；衡量基因子集 $S$ 在 $64$ 维流形中张开的可控椭球体积 |
| 次模鸡尾酒选择（$(1-1/e)$ 保证） | $\log\det W_c(S)$ 是正规化、单调增次模函数，贪心算法必达全局最优至少 $1 - 1/e \approx 63.2\%$ 体积收益 |
| 方向最小控制能量 $E_{\min}(j)$（Eq. 5） | 向目标差 $\Delta z = z_{\mathrm{tgt}} - A^H z_{\mathrm{src}}$ 所需最小能量与 $(\Delta z^\top C_H b_j)^2$ 成反比 |
| 凝聚态 FISTA-AL 实时优化（Prop. 3） | 通过 Toeplitz 转移矩阵 $M_z, G_u$ 消去中间状态约束，仅在 $u\in[0,u_{\max}]^{Hk}$ 上做闭式加速近端梯度 + 对偶上升迭代 |
| 鲁棒不变集 RPI + 延迟补偿 + Practical ISS（Prop. 4, Alg. 2, Thm. 1） | 由校准残差 $w_{\mathrm{base}}$ 与辅助律 $K_{\mathrm{DARE}}$ 构成收紧管边界 $\delta_{\mathrm{nom},t} = \delta_{\mathrm{safe}} - r_t$，在名义可行且有界失配/延迟下由三步 Lyapunov 证明递归可行性与 Practical ISS |

---

## 3. 核心定理物理意义与顶刊级实测“排雷/洞察”

### 3.1 为什么非正规度（Kreiss Gain）与 `Stiefel` 正交参数化缺一不可？（Fig. 3 & Sensitivity Sweep）
这是线性系统理论中极其关键的概念——**谱半径 $\rho(A)$ 只决定 $t\to\infty$ 的渐近稳定性，有限步瞬态尖峰由奇异值 $\|A\|_2$、特征向量条件数 $\kappa(V)$ 与 Kreiss 常数 $\mathcal{K}(A)$ 支配**！
- **无约束 EDMD**：虽然特征值只有 $\rho = 0.0533$，但特征向量矩阵病态（$\kappa(V)=73.40$，$\sigma_{\max}=0.1892$），叠加未受控执行器映射，导致 $20$ 步滚动范数跳升至 **`3.224`**（KGC 的 **$8.06\times$**）；
- **去掉 Stiefel 约束的 `KGC-Unconstrained A` (`No-Stiefel`)**：实测直接暴露出隐蔽失效——不仅 $\sigma_{\max}(A)$ 突破 $1$ 升至 **`1.0184`**（$\rho=0.9990$，$\kappa(V)=106.30$，Kreiss 下界 $\mathcal{K}_{\mathrm{lb}}=1.0035>1$，`rollout20=0.463`），而且**双组合零样本外推 `combo_seen0` 直接跌至 `-0.005`**！
- **KGC-MPC（`Stiefel` 极投影）**：硬约束 $\|A\|_2 \le \gamma_{\mathrm{schur}}$ 让实际 $\sigma_{\max}=0.8897 < 1$、$\mathcal{K}_{\mathrm{lb}}=1.000$、$20$ 步平稳收敛于 **`0.400`**；超参敏感性扫描 $\gamma_{\mathrm{schur}}\in[0.80, 0.99]$（`seen2 = 0.938–0.941`）与双线性秩 $q\in[0, 24]$（`0.935–0.941`）均保持稳健。

### 3.2 统一 FISTA-AL 名义安全前沿与跨种子失配对比（Table III & Fig. 4a）
- **零违约名义前沿**：旧版本 `safety_mpc_pareto` 使用惩罚梯度下降会产生偶发违约；升级后全面统一调用 `CondensedSafeMPCSolver` (`FISTA-AL`)，在全部 $\delta_{\mathrm{safe}}\in[0.05,0.85]$ 上达到 **`0.0%` 违约率**，$\delta_{\mathrm{safe}}=0.05$ 时最大脱靶漂移从 `Box-LQR` 的 `0.2177` 降至 `0.0500`（**`-77.0%`**）。
- **跨种子（`seed=1`）非线性双线性被控对象失配**：`KGC Tube-MPC` 全程 **`0.0%` 违约**，$\delta=0.05$ 时脱靶漂移仅为 **`0.0275`**（较 `Box-LQR` `0.1247` 降低 **`-78.0%`**）；$\delta=0.25$ 时拿下**最低靶点误差 `0.696`**（优于 `CL-MPC` `0.773`、`OL-MPC` `0.751` 和 `Box-LQR` `0.799`）。

### 3.3 实时求解器对决与 $N=20$ 靶点全组件统计显著性消融（Table IV & Fig. 6）
- **CPU BLAS vs Conic 求解器 & A100 批处理交叉点 $B^\star=8$**：
  - 热启动 CPU BLAS `FISTA-AL` 从 $H=4$ 到 $H=32$ 仅需 **`4.45 -> 18.61 ms/step`**（$H=8$ `6.33 ms`，比冷启动 `22.45 ms` 快 `17.1%`），与精确二阶锥内点法 `CVXPY-Clarabel SOCP` 最优目标值差距仅 **`0.17%–0.30%`**，但在 $H=32$ 上比 `Clarabel`（`2063.1 ms`）快 **`110.9×`**、比 `CVXPY-SCS ADMM`（`307.74 ms`）快 **`16.5×`**、比 `Autograd-Adam`（`4762.4 ms`，`0.0%` 满足率）快 **`255.9×`**！
  - 在共享 `A100-SXM4-80GB` 上（`exclusive_gpu_verified: false`，`procs=2`），单轨迹 $B=1$ 因 CUDA 启动开销耗时 `41.16 ms`，但批量并行药筛时从 $B^\star=8$（`5.24 ms/traj`）反超 CPU，并在 $B=64$ 达 **`0.671 ms/traj`**、$B=256$ 达 **`0.168 ms/traj`**（间隙稳定 `0.28%`）。
- **$N=20$ 靶点 $\times$ 12 压测工况（240 幕/控制器）+ Wilcoxon BH-FDR 检验**：
  - 完整 **`KGC Tube-MPC (RPI+Delay)`** 在全部 240 幕强失配（$\kappa\le 5$）、强饱和（$u_{\max}\ge 0.25$）、多步延迟（$\tau\le 3$）中取得 **`100.0%` 递推可行性与 `100.0%` 安全管满足率**（总归一化漂移 `0.425`）。
  - **真消融定责**：
    - 去掉 **RPI 收紧**（`w/o RPI`）$\implies$ $\kappa=1$ 满足率跌至 `90.0%`（$q=2.29\times 10^{-4}$），$\kappa=5$ 跌至 `95.4%`（`NormOff=0.703`, $q=3.37\times 10^{-6}$），全工况汇总 $q_{\mathrm{off}}=1.80\times 10^{-39}$；
    - 去掉 **延迟预测**（`w/o Delay-Pred`）$\implies$ $\tau=3$ 时漂移大增 **`+55.3%`**（`0.531 vs 0.342`, $q_{\mathrm{off}}=6.36\times 10^{-6}$）；
    - 去掉 **辅助 DARE 反馈**（`w/o Ancillary`）$\implies$ 控制器不再纠偏而“滑行”，因而漂移略低（`0.386 vs 0.425`，$q_{\mathrm{off}}=8.79\times 10^{-32}$），但跟踪误差相应升高（`0.904–0.929`）；
    - **`Box-LQR`** 全工况崩盘：汇总满足率仅 **`70.2%`**（$q_{\mathrm{sat}}=3.41\times 10^{-25}$），$\kappa=5$ 降至 `57.1%` 且归一化漂移达 **`53.7`**。

---

## 4. 代码与实测数据对照表

| 组件 | 路径 | 关键实现 |
|---|---|---|
| 核心控制算法 | [src/kgc_mpc.py](src/kgc_mpc.py) | `polar_orthogonal`, `KGCSystem` (含 `stiefel_contraction`), `compute_nonnormal_diagnostics`, `CondensedSafeMPCSolver` (`fista_al`/`adam_penalty`), `TubeSafeMPCController` (`use_ancillary`/`use_safety_guard`), `select_cocktail_submodular` |
| TCST 实时与压测基准 | [src/run_tcst_benchmarks.py](src/run_tcst_benchmarks.py) | `Clarabel`/`SCS` 对比、CPU BLAS vs A100 $B=1..256$、$N=20$ 靶点 $\times$ 12 工况全组件消融 + Wilcoxon BH-FDR、Kreiss 与超参灵敏度 |
| 基线对照 | [src/baselines.py](src/baselines.py) | `UnconstrainedEDMD`, `CPALatentAE`, `GEARSCoExprGNN`, `compute_linear_additive_baseline` |
| 流式大数据加载 | [src/data_loader.py](src/data_loader.py) | C 引擎分块稀疏加载 `Sci-Plex3` (762,795) 与无表头适配 `Sci-Plex4` (98,397) |
| 全套一致性检验 | [src/verify_paper.py](src/verify_paper.py) | 核对全部 `weights/*.npz` ($\sigma_{\max}(A)\le 0.955$)、18 份 CSV/JSON 产物及 `10 pages`、`0 Overfull` |

---

## 5. 投稿选刊策略与上交 816 高频追问

- **投稿组合建议**：首选冲刺控制工程顶刊 **IEEE TCST**（完整对齐 10 页标准双栏、Theorem 1 / Algorithm 2、Conic 对比与 240 幕统计验证），理论风格加厚版可投 **Automatica** 或 **IEEE TCYB**；与对偶观测侧 Paper 5 (*GRAM-UIO*) 在摘要、问题定义与数据实验上完全正交独立。

1. **Q1：为什么去掉 `Ancillary DARE` 反馈（`w/o Ancillary`）后归一化漂移（`0.386`）反而比完整 `Tube-MPC`（`0.425`）低？是不是辅助反馈没用？**
   - **答**：恰好相反！这在报告和论文中都是极其严谨的控制消融物理直觉：当存在较大模型失配（$\kappa \le 5$）时，辅助反馈 $u_{\mathrm{anc}} = K_{\mathrm{DARE}}(\hat{z}_t - \bar{z}_t)$ 会主动耗费能量把失配偏移的细胞状态“拉回”标称参考轨迹以降低**主目标跟踪误差**；去掉 `K_DARE` 后控制器退化为近似开环/保守 MPC，“不再用力对抗失配”自然侧向扰动小了一点（`0.386 vs 0.425`），代价是主目标跟踪误差升高（`0.904–0.929` vs `0.873–0.928`）。两者清晰揭示了鲁棒跟踪与控制温和度之间的权衡。
2. **Q2：Theorem 1 的 Practical ISS 证明（附录 A）是怎么串起来的？**
   - **答**：分三步严密闭环：(i) **递归可行性**：通过由校准不确定集外推的管半径 $r_t$ 收紧名义约束至 $\delta_{\mathrm{safe}} - r_t$，使得上一步最优控制序列的移位后缀配零终端输入仍是下一步可行解；(ii) **状态有界管不变性**：利用误差动力学 $e_{t+1} = (A + B_S K_{\mathrm{DARE}})e_t + w_t$ 在 Schur 闭环谱半径 $<1$ 下保持 $\|P_\perp e_t\|_2 \le r_t$，加上名义约束三角形不等式即可保真实状态永不越界 $\delta_{\mathrm{safe}}$；(iii) **Lyapunov Practical ISS**：构造 DARE 值函数 $V(\bar{z}) = \|\bar{z} - z_{\mathrm{eq}}\|_{P_{\mathrm{DARE}}}^2$，由最优值下降减去最终稳态扰动常数球，证明状态指数收敛至平衡点邻域。
3. **Q3：把 Paper 3（KGC-MPC）与 Paper 5（GRAM-UIO）放在一起看，核心控制思想是什么？**
   - **答**：**卡尔曼能控性–能观性对偶原则（Kalman Controllability–Observability Duality）**：Paper 5 通过正交**能观性 Gramian** 优化“观测侧有限 $p$-基因面板”，并在未知医院扰动下重建状态 $\hat{z}_k$；Paper 3 通过**能控性 Gramian** 与收缩 Koopman 模型优化“执行侧协同药物/基因干预 $u_k$”，并在安全管与 RPI 不变集约束下将其闭环推进目标态 $z_{\mathrm{tgt}}$。两篇在次模 $\log\det$ 理论、Riccati DARE 与系统安全性/鲁棒性上形成对偶闭环。
