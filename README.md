# AI 驱动药物虚拟筛选工作流

> 一套面向新靶点的**端到端、可复现、带阳性对照验证**的药物虚拟筛选范式。从靶点立项到实验验证，主干清晰、AI 分支边界明确。

---

## 这是什么

本仓库把"从一个蛋白靶点出发，到产出可送实验的候选分子"的完整计算流程，拆解为 **12 篇可逐篇照做的方法文档**。核心特点：

- **主干 + 分支分离**：主干是"靶点→结构→口袋→分子库→筛选→可开发性门→Ensemble 对接→MD→实验"的必走链路；两个自研 AI 模型作为**分支**嵌入，明确其价值与局限，不喧宾夺主。
- **方法论优先**：每一步讲清"为什么做、解决什么问题、不做会怎样"，附可执行命令与验收标准，不依赖特定项目案例。
- **诚实分层**：不夸大 AI，明确"对接分≠真实结合能、AI 预测不能代替动力学和实验"；可靠性来自多类证据串联。
- **可复现**：参数冻结在先、结果产出在后，每步有自检清单。

---

## 文档结构

| 编号 | 文档 | 核心内容 |
|---|---|---|
| 00 | [工作流总览](00_工作流总览.md) | 整体架构、AI 价值定位、适用范围 |
| 01 | [靶点选取与立项](01_靶点选取与立项.md) | 疾病→通路→靶点的证据链（L1–L5）、可成药性评估 |
| 02 | [靶点结构建模](02_靶点结构建模.md) | 实验结构/预测结构获取、pLDDT 检查、预处理 |
| 03 | [结合口袋识别与确认](03_结合口袋识别与确认.md) | 无偏扫描 + 文献锚点 + 能量分解的三重证据法 |
| 04 | [分子库构建与分轨](04_分子库构建与分轨.md) | 库来源、清洗规则、按化学空间分轨 |
| 05 | [多线并行虚拟筛选](05_多线并行虚拟筛选.md) | 对接协议验证、双线引擎、PoseBusters 物理门 |
| 06 | [分支一：亲和力预测模型（AffinX）](06_分支一_亲和力预测模型.md) | XGBoost + Δ-learning + CASF 外部验证 |
| 07 | [分支二：生成式分子模型（MolGPT-LoRA）](07_分支二_生成式分子模型.md) | LoRA 轻量微调、四关卡质检、同闸汇入 |
| 08 | [多证据融合与可开发性门](08_多证据融合与可开发性门.md) | 秩融合、四维可开发性门、短名单冻结 |
| 09 | [Ensemble 对接与 MD 终裁](09_Ensemble对接与MD终裁.md) | 多 seed ensemble、短 MD 机械裁决、算力闸门 |
| 10 | [实验验证闭环](10_实验验证闭环.md) | 四级湿实验验证、计算与实验互相印证 |
| 12 | [工作流自检清单](12_工作流自检清单.md) | 九阶段打勾表，实际开干时对照使用 |

---

## 两个自研 AI 模型

本工作流包含两个可独立使用的开源模型，作为分支嵌入主干，也可单独集成到你的管线中。

### 1. AffinX — 亲和力预测模型

- **定位**：对通过物理质检的对接构象预测结合强度（pKd），作为多证据融合中的独立排序一票。
- **方法**：XGBoost + Δ-learning（学习对接打分的残差修正），特征融合 2D 描述符、Morgan 指纹、口袋特征、PLIP 相互作用特征与对接能量分项。
- **外部验证**：在 CASF-2016 基准集上 Pearson 相关系数 **R = 0.8185**。
- **安全机制**：骨架分组审计量化乐观偏差；每个预测附带适用域标签（Tanimoto 最近邻）与按距离放大的 95% 预测区间；**只排序、不硬淘汰**，域外分子以物理证据为主。
- **详见**：[06 分支一](06_分支一_亲和力预测模型.md) ｜ [模型卡与权重](models/affinx/README.md)

### 2. MolGPT-LoRA — 靶点特异性分子生成

- **定位**：用少量靶点已知配体对预训练生成模型做轻量领域适配，扩展可探索的化学空间。
- **方法**：冻结 MolGPT 基座全部参数，仅注入低秩适配矩阵（LoRA），可训练参数占比极小，单卡数分钟即可为新靶点训练一个适配插件。
- **质检**：生成产物经合法性、新颖性、领域相关性、类药性/PAINS 四道关卡 + 结构合理性审查，**与主库完全同闸汇入，无任何优待**。
- **对照实验**：强制做"微调前 vs 微调后"等量生成对比，用数据而非口号证明微调价值。
- **详见**：[07 分支二](07_分支二_生成式分子模型.md) ｜ [模型卡与权重](models/molgpt_lora/README.md)

> 两个模型的**训练代码、训练好的权重与模型卡**均已随本仓库发布（见 [`models/affinx/`](models/affinx/) 与 [`models/molgpt_lora/`](models/molgpt_lora/)）。它们是**锦上添花的分支**——即使不用，主干用"结构对接 + 物理质检 + 可开发性门 + 动力学 + 实验"依然能产出可靠候选。

---

## 适用范围

- 有（或能获取）三维结构、具备一定可成药性线索的蛋白靶点；
- 最好有一个已知活性分子作为阳性对照锚点（用于事后验证流程，不进训练集）；
- 计算资源：CPU 可完成大部分环节；Ensemble 对接与 MD 推荐 GPU。

---

## 如何使用

1. **第一次读**：按 00 → 10 顺序通读，建立全局观。
2. **实际开干**：把 [12 自检清单](12_工作流自检清单.md) 打出来，每完成一步对照打勾。
3. **遇到 AI 模型**：先读 06/07 的"边界"章节，明确什么时候该用、什么时候不该信。
4. **直接跑管线**：主干计算管线脚本见 [`pipeline/`](pipeline/)，按编号顺序即可从结构预处理一路跑到 MD 终裁分析。

---

## 开源工具与依赖

本工作流大量使用成熟开源工具，以下是各环节主要依赖及其仓库：

| 环节 | 工具 | 用途 | GitHub |
|---|---|---|---|
| 结构预处理 | **PDBFixer** | 补缺失原子、加氢、去杂 | https://github.com/openmm/pdbfixer |
| 结构预处理 | **RDKit** | 分子处理、描述符、指纹 | https://github.com/rdkit/rdkit |
| 口袋识别 | **FPocket** | 无偏口袋扫描 | https://github.com/Discngine/fpocket |
| 口袋识别 | **CASTp** | 基于 alpha-shape 的口袋识别 | https://github.com/WeiShao-Lab/CASTp |
| AI 初筛（轻线） | **DrugCLIP** | 蛋白-配体对比学习打分 | https://github.com/bowen-gao/DrugCLIP |
| 物理对接（重线） | **AutoDock-GPU** | GPU 加速对接 | https://github.com/ccsb-scripps/AutoDock-GPU |
| 物理对接 | **AutoDock Vina** | 经典对接引擎 | https://github.com/ccsb-scripps/AutoDock-Vina |
| 配体参数化 | **Meeko** | PDBQT 准备 | https://github.com/forlilab/Meeko |
| 物理规则门 | **PoseBusters** | 构象物理合理性检查 | https://github.com/maabuu/posebusters |
| 相互作用分析 | **PLIP** | 蛋白-配体相互作用特征 | https://github.com/chemosim-lab/PLIP |
| 分子动力学 | **GROMACS** | MD 模拟 | https://github.com/gromacs/gromacs |
| 结合自由能 | **gmx_MMPBSA** | MM/PBSA 计算 | https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA |
| 亲和力模型 | **XGBoost** | AffinX 后端 | https://github.com/dmlc/xgboost |
| 生成式基座 | **MolGPT** | 分子生成 Transformer | Bagal et al., 2022, *JCIM* |

两个自研模型（AffinX、MolGPT-LoRA）的代码与权重见本仓库 [`models/`](models/) 目录。

---

## 许可

本项目以 [MIT 协议](LICENSE) 开源。两个自研模型（AffinX、MolGPT-LoRA）的权重与训练代码同样以 MIT 协议发布，欢迎集成与二次开发。

---

## 仓库结构

```
├── README.md                  # 本文件
├── LICENSE                    # MIT 协议
├── 00–10、12_*.md             # 工作流方法文档（按编号阅读）
├── models/
│   ├── affinx/                # 分支一：亲和力预测模型
│   │   ├── README.md          # 模型卡（性能、特征、适用域、复现方式）
│   │   ├── xgboost_optimized_v3.pkl       # 训练好的模型权重
│   │   ├── train_fingerprints_13770.npz   # 适用域参考指纹库
│   │   ├── best_params.json               # 最优超参数
│   │   └── code/              # 训练 / 推理 / 适用域代码
│   └── molgpt_lora/           # 分支二：生成式分子模型
│       ├── README.md          # 模型卡（LoRA 配置、质检流程、加载方式）
│       ├── adapter/           # 训练好的 LoRA 适配器权重
│       └── code/              # 数据构建 / 微调 / 生成对比 / 质检代码
└── pipeline/                  # 主干计算管线脚本（按编号顺序运行，与方法文档一一对应）
    ├── README.md              # 管线总览：运行顺序、依赖安装、文档映射
    ├── requirements.txt       # Python 依赖清单
    ├── 01_prepare_structure.py   # 结构预处理 + pLDDT 质量报告（对应 02 篇）
    ├── 02_identify_pockets.py    # 口袋识别与对接盒定义（对应 03 篇）
    ├── 03_validate_protocol.py   # 筛选前协议验证：redocking / 阳性锚点（对应 05 篇）
    ├── 04_prepare_library.py     # 分子库清洗与分轨（对应 04 篇）
    ├── 05_dock_screen.py         # Vina 批量对接与第一门（对应 05 篇）
    ├── 06_posebusters_qc.py      # 构象物理规则质检（对应 05 篇）
    ├── 07_fusion_rank.py         # 多证据秩融合与可开发性提醒（对应 08 篇）
    └── 08_md_analysis.py         # MD 轨迹 B1–B4 机械裁决（对应 09 篇）
```
