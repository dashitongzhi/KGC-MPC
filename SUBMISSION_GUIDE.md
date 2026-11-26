# Paper 3 (`KGC-MPC`) 投递操作与期刊战略指南

> **现状总览**：除个人实名信息外，技术审稿与开源仓全套材料已完备。  
> - **公开代码仓**：[`https://github.com/dashitongzhi/KGC-MPC`](https://github.com/dashitongzhi/KGC-MPC)  
> - **10 页 `IEEEtran` 终稿**：[`KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf`](KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf)  
> - **英文投稿信**：[`submission/COVER_LETTER_IEEE.md`](submission/COVER_LETTER_IEEE.md)  
> - **投稿系统复制粘贴库**：[`submission/DECLARATIONS_AND_METADATA.md`](submission/DECLARATIONS_AND_METADATA.md)

---

## 一、投前 3 分钟个人填空清单（仅余这些 `TODO(PERSONAL_INFO)`）

在点击期刊系统 `Submit` 之前，全局搜索 `TODO(PERSONAL_INFO)`，只需替换以下 3 处个人信息：

1. **[`manuscript/main.tex`](manuscript/main.tex)**：
   - 第 `30–32` 行 `\author` 与 `\thanks`：填写真实 `[Author Full Name]`、`[Department / School]`、`[University / Institution]`、`[City] [Postal Code], [Country]`、`[author@institution.edu]`、`[0000-0000-0000-0000]` 及基金号（无基金直接删掉基金那句 `\thanks`）。
   - 第 `35–36` 行 `\markboth`：把 `[Author Surname]` 改为第一作者姓氏（若改投 *Automatica* 或 *IEEE TCYB*，同步修改期刊英文名）。
   - 第 `315–317` 行（可选）：若目标刊要求初始投稿附带作者简介，取消注释 `IEEEbiography` 并填入个人简历与证件照。
2. **[`submission/COVER_LETTER_IEEE.md`](submission/COVER_LETTER_IEEE.md)**：
   - 替换开头日期 `[TODO(PERSONAL_INFO): Date]` 与文末通讯作者署名、单位、地址、邮箱和 ORCID。
3. **[`submission/DECLARATIONS_AND_METADATA.md`](submission/DECLARATIONS_AND_METADATA.md)**：
   - 确认基金资助声明（`Funding Statement`）与 CRediT 作者贡献列表。

改完上述 3 处后，本地执行一次 PDF 重编译、一致性校验并推送更新即可直接上传投稿系统：
```sh
cd manuscript && tectonic --keep-logs main.tex && cp main.pdf ../KGC-MPC_Koopman_Perturbation_Control_IEEE.pdf && cd ..
python3 -B src/verify_paper.py
git add -A && git commit -m "chore: populate author and affiliation metadata" && git push origin main
```

---

## 二、分层期刊投递策略与工作区投稿顺序

| 优先级 | 目标期刊 | 定位与适配度 | 模板 / 页数限制 | 初审与送审要点 |
|---|---|---|---|---|
| **首选主冲 (1st)** | ***IEEE Transactions on Control Systems Technology* (TCST)** | 现代控制理论工程落地与求解器实时性验证旗舰顶刊，极为看重真实建模对比、硬件计算开销及物理鲁棒性 | `IEEEtran` 双栏；常规论文通常 10–12 页含超页费窗口（当前 **10 页**，`0 Overfull \hbox`） | Cover Letter 首推 `Prop. 1` Stiefel 抑制 Kreiss 瞬态放大（$8.06\times$）、`Condensed FISTA-AL` 对 `Clarabel SOCP`/`SCS ADMM` 加速 **$110.9\times / 16.5\times$**（$B^\star=8$ GPU 交叉点）、以及 **Theorem 1 / Algorithm 2** 在 240 幕压测下的 `100.0%` 递推可行与 BH-FDR 检验 |
| **理论备选 (2nd)** | ***Automatica* (Elsevier)** | 控制理论最高殿堂之一，高度青睐“Stiefel 收缩 + Submodular Gramian + Lyapunov Practical ISS 三段式证明” | 切换至 `elsarticle` / `autart` 模板，并使用 `DECLARATIONS_AND_METADATA.md` 中 `<= 200` 词精简摘要与 5 条 `<= 85` 字符 `Highlights` | 强调无约束 `No-Stiefel` 的 $\sigma_{\max}=1.0184$ 反例、Theorem 1 完整 Lyapunov 值函数下降界及全组件消融 |
| **交叉顶刊 (3rd)** | ***IEEE Transactions on Cybernetics* (TCYB)** | 控制论与大规模复杂生物/信息系统交叉顶刊 | `IEEEtran` 双栏，当前 **10 页**零改造成本直接投稿 | 突出 `1,021,000` 个真实单细胞四队列跨基因-化学扰动验证与 $762{,}795$ 细胞 Sci-Plex3 全线最低 MSE |
| **同系备选 (4th)** | ***IEEE TCNS / IEEE TNNLS*** | 网络化控制与非线性学习系统 | `IEEEtran` 双栏 | 聚焦 Koopman 双线性算子学习与安全约束优化加速 |

### 与工作区其余论文的投稿区隔与节奏建议
1. **与对偶观测侧 Paper 5 (`GRAM-UIO`) 的严格正交关系**：Paper 3 专注**“执行控制侧”**——用**能控性 Gramian**、Schur-Stiefel Koopman 动力学与安全管 Tube-MPC 设计多药物/基因干预剂量（数据集：`Norman`, `Adamson`, `Sci-Plex3/4` 共 `1,021,000` 细胞）；Paper 5 专注**“状态观测侧”**——用能观性 Gramian 与 UIO 重建未知批效应下的细胞状态（数据集：`GSE194122` 多院区 CITE-seq）。两者问题定义、理论工具、代码仓与实验队列完全独立，`0` 重合图表与公式，可并行独立投稿。
2. **与其他论文区隔**：Paper 1/2 聚焦多组学表征与闭环聚类，Paper 4 聚焦空间多智能体图牵制，Paper 6 聚焦随机薛定谔桥概率传输，Paper 7/8 聚焦 Port-Hamiltonian 收缩与对偶因果主动干预，均无一稿多投或审稿池冲突。

---

## 三、高频审稿质疑预判与直接回复口径

1. **Q：为什么去掉 `Ancillary DARE` 反馈（`w/o Ancillary`）后，汇总归一化漂移 `0.386` 略低于完整 `Tube-MPC`（`0.425`）？**  
   - *口径*：我们在正文中直接剖析了这一物理机制——在严重模型失配（$\kappa \le 5$）下，辅助反馈 $K_{\mathrm{DARE}}(\hat{z}_t - \bar{z}_t)$ 会主动耗能把偏移状态“拉回”参考轨道以降低**主目标跟踪误差**；去掉 `K_DARE` 后控制器退化为不再对抗失配的保守滑行策略，自然少产生一部分纠偏控制量，代价是主目标跟踪误差上升（`0.904–0.929` vs.\ `0.873–0.928`）。两者清晰揭示了跟踪性能与控制温和度之间的帕累托权衡。
2. **Q：为什么在 Norman `unseen_single` 上 `KGC-MPC`（`0.201`）低于 `GEARS-CoExpr`（`0.477`），以及在 Sci-Plex4 `MCF7` 上低于 1-step `CPA`（`0.0352` vs `0.0192`）？**  
   - *口径*：这是本工作的**坦诚边界（Candid Boundary）**，已在正文 Section V & VI 明确说明。对于训练集中从未受直接激励的单基因列，纯线性共表达先验经 $r=64$ 收缩瓶颈映射的容量低于两层非收缩 GNN（`GEARS-CoExpr`）；而在高度聚焦单一通路的 `Adamson` 上 `KGC-MPC` 取得 `0.728`（胜 `KGC-Linear +0.048` 并将 `EDMD 0.000` / `CPA -0.027` 远远甩开）。同理，`MCF7` 双药单步预测提示我们后续可引入细胞系自适应秩正则化。
3. **Q：为什么论文中 GPU 批处理测时要标注共享 A100 与 $B^\star=8$ 交叉点？**  
   - *口径*：我们在 `results/environment_tcst.json` 与正文中如实报告运行于共享 A100（`exclusive_gpu_verified: false`, `active_compute_procs_at_finish: 2`）。单轨（$B=1$）因 CUDA kernel 启动开销耗时 `41.16 ms`，因此单病例在线反馈优先采用 **CPU BLAS**（`4.45–18.61 ms/step`）；而高通量离线虚拟筛选在 $B \ge B^\star = 8$ 时 GPU 并行摊薄开销后即可反超 CPU，并在 $B=256$ 达到 `0.168 ms/traj`。
4. **Q：闭环控制为什么在跨随机种子（`seed=1`）非线性双线性被控对象上检验，而不是活细胞实时反馈仪？**  
   - *口径*：当前单细胞 Perturb-seq / Sci-Plex 均属于破坏性终点测序，物理上无法对同一个活细胞进行连续 $H$ 步闭环流式给药再测序。因此采用在真实 `91,205` Norman 单细胞上独立随机种子拟合的含双线性交互与高倍失配的非线性孪生被控对象（`seed=1`）作为真实失配植物，这是单细胞控制系统验证的标准且诚实的做法，并在第 VI 节展望了结合纵向活细胞成像（live-cell imaging）的下一步验证。
