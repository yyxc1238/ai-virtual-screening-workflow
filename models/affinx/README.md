# AffinX — 蛋白-配体结合强度（pKd）预测模型 · XGBoost v3

模型卡（Model Card）。AffinX 是一个 Δ-learning 风格的蛋白-配体结合强度回归模型：
以配体 2D 结构特征为基础，叠加复合物 3D/接触特征与分子对接能量项，输出 pKd
（= −log10 Kd）及换算 IC50(μM)，并对每条预测附**适用域标记**与 **95% 近似预测区间**，
提示"这条预测落在训练化学空间内吗、该信几分"。

## 1. 目录内容

| 文件 | 说明 |
|---|---|
| `xgboost_optimized_v3.pkl` | 正式权重（约 1.8 MB；pickle dict：`model` / `feature_cols` / `params`） |
| `train_fingerprints_13770.npz` | 适用域参考库：13,770 条训练池配体的 Morgan 指纹（packbits 压缩，含 PDB ID / pKd / split） |
| `best_params.json` | Optuna 搜索得到的最优超参数（与 pkl 内 `params` 一致） |
| `code/` | 训练 / 评估 / 推理 / 适用域全流程脚本（见 §5） |

## 2. 基本信息

| 项目 | 内容 |
|---|---|
| 算法 | XGBoost 梯度提升树回归（`reg:squarederror`，`tree_method=hist`） |
| 任务 | 蛋白-配体结合强度回归：pKd 预测，换算 IC50(μM) = 10^(−pKd) × 10^6 |
| 训练样本 | PDBBind v2020 特征表 14,055 条；排除 CASF-2016 core set 285 个外部基准后训练池 **13,770** 条 |
| 特征维度 | **2,110** = 18 个 2D 描述符 + 2,048 bit Morgan 指纹(r=2) + 蛋白/口袋 3 维 + PLIP 接触/3D 维 + 6 个 Vina 能量项 |
| 调参 | Optuna 50 trials，train-CV gap 惩罚 0.3；固定种子 `random_state=42` |

最优超参数（与 `best_params.json` 一致）：
`n_estimators=700, max_depth=6, learning_rate=0.04662, subsample=0.5293, colsample_bytree=0.5578, min_child_weight=12, reg_alpha=0.505, reg_lambda=21.177`

## 3. 性能指标

| 指标 | 数值 | 说明 |
|---|---|---|
| 5 折 CV R² | **0.5629 ± 0.0161** | 折间方差小，稳定性可接受 |
| CV RMSE / MAE | 1.2318 / 0.9530 | pKd 对数单位 |
| 训练 R² | 0.7431 | train-CV gap = 0.1802，通过 gap 惩罚压制过拟合 |
| 配体-only 基线 R² | 0.4919 | 叠加 3D/对接特征后 **+0.0710**，复合物特征有增量贡献 |
| CASF-2016 core set | **R = 0.8185，R² = 0.6177，RMSE = 1.3420** | 285 复合物独立基准，未参与训练/调参 |

**骨架泄露审计**（`code/audit_scaffold_split.py`，补充评估、未替换正式权重）：
13,767 个有效样本、6,796 个 Bemis-Murcko 独立骨架；同样本、同特征、同冻结超参下
随机 5 折 R² = 0.559 ± 0.010 vs 骨架分组 5 折 R² = 0.503 ± 0.033，落差 0.057
（中等随机划分乐观偏差）。结论：骨架外推下模型仍保有 R²≈0.50 的真实排序力；
下一代模型建议直接采用 scaffold split 训练。

## 4. 数据来源与许可

- 训练数据来自 **PDBBind v2020.R1**（INDEX_general_PL 口径；Ki/Kd/IC50 统一换算
  pKd，不等式样本按其 "use" 策略处理）。请从官方网站
  **https://www.pdbbind-plus.org.cn** 申请获取，本仓库**不再分发**任何 PDBBind 原始数据。
- CASF-2016 core set（285 复合物）仅作完全外部基准，训练/调参全程排除。
- 本仓库不含任何自建库分子；`train_fingerprints_13770.npz` 仅存储 PDBBind 训练池
  配体的指纹与 PDB ID，用于适用域最近邻检索。

## 5. 脚本说明（code/）

| 脚本 | 作用 |
|---|---|
| `predict.py` | 单表推理入口：SMILES → pKd / IC50 / 适用域 / 95% 区间 |
| `screen.py` | 分子库批量快筛与排序（域内优先） |
| `applicability_domain.py` | 适用域（AD）与不确定性区间实现 |
| `train.py` | 正式训练编排入口（step1 → step4，需 PDBBind 数据） |
| `step1_load_data.py` | 解析 PDBBind 索引，构建样本表 |
| `step2_extract_features.py` | 2D 描述符 + Morgan 指纹 + 蛋白/口袋特征提取 |
| `step3_train_baseline.py` | baseline 模型训练与可视化 |
| `step4_optimized_v3.py` | v3 正式训练（Optuna 50 trials + gap 惩罚） |
| `step5_evaluate_lite.py` | 轻量评估：偏置检测 / 标签分布 / 特征重要性 |
| `build_train_fp_ref.py` | 重建适用域参考库 npz |
| `audit_scaffold_split.py` | Murcko 骨架泄露审计 |
| `train_demo.py` | 小规模教学闭环（合成数据抽样，无需 PDBBind） |
| `config.py` | 路径与全局常量 |

依赖见 `code/requirements.txt`。

## 6. 如何加载使用

### 6.1 命令行（推荐）

```bash
cd code

# 单表预测：输入 CSV 需含 mol_id, smiles 两列；可选 vina_docking_score 列
# 蛋白/口袋残基数建议提供真实值（输入同名列 > 命令行参数 > 0+警告）
python predict.py --input candidates.csv --output results.csv \
    --prot_seq_len 400 --prot_n_residues 400 --pocket_n_residues 20

# 分子库批量快筛（域内优先排序）
python screen.py --input library.csv --topk 50

# 无需 PDBBind 的教学闭环（合成数据抽样演示训练流程）
python train_demo.py
```

### 6.2 Python 直接加载权重

```python
import pickle
import numpy as np

with open("xgboost_optimized_v3.pkl", "rb") as f:
    pkg = pickle.load(f)

model = pkg["model"]                 # xgboost.XGBRegressor
feature_cols = pkg["feature_cols"]   # 2,110 个特征名（有序）
params = pkg["params"]               # 最优超参

# 特征向量需严格按 feature_cols 顺序拼装；2D 部分（18 描述符 + 2048 Morgan 位）
# 可复用 code/predict.py 的 compute_2d_features(smiles) 生成（特征命名与训练一致）
X = np.zeros((1, len(feature_cols)), dtype=np.float32)
# ... 填充特征 ...
pkd = float(model.predict(X)[0])
ic50_uM = 10 ** (-pkd) * 1e6
print(pkd, ic50_uM)
```

> 推理口径说明：`predict.py` / `screen.py` 为 **"2D+快筛"近似口径**（2D 特征由
> SMILES 实时计算，vina_total 取输入表的对接分列、无则填 0，其余 3D/PLIP 特征
> 填 0）。如需全特征口径，请自行准备对接/接触特征列并按 `feature_cols` 拼装
> （原管线中负责全特征合并的编排脚本未随本仓库发布）。

## 7. 适用域（Applicability Domain）与不确定性区间

- **结构表征**：与模型输入指纹同源——2,048 bit Morgan（radius=2，RDKit
  MorganGenerator，含氢分子）；相似度用 Tanimoto（Jaccard）。
- **参考库**：`train_fingerprints_13770.npz`（13,770 条训练池配体指纹，np.packbits
  存储）；由 `code/build_train_fp_ref.py` 从全量特征表重建。
- **分档阈值**（事前冻结，不随结果回调）：

  | max Tanimoto（最近邻） | 标记 | σ 放大倍数 |
  |---|---|---|
  | ≥ 0.50 | IN_DOMAIN | 1.0 |
  | 0.30 – 0.50 | BORDERLINE | 1.0 → 1.5 线性 |
  | < 0.30 | OUT_OF_DOMAIN | 1.5 → 2.0 线性 |

- **95% 近似区间**：σ₀ = 五折 CV RMSE = **1.2318 pKd**；σ(s) 按上表随最近邻距离
  线性放大；区间 = pKd ± 1.96·σ(s)（高斯残差近似）。
- **这是工程近似，不是校准后的条件分布**：未做残差随化学空间变化的分位数/异方差
  校准；σ 放大规则为透明可复核的线性启发式。IN_DOMAIN 区间宽度约 ±2.4 pKd，
  反映 PDBBind 亲和力预测任务本身的固有噪声（CASF RMSE 1.34）。
- **数值实现说明**：部分环境（numpy 2.x + OpenBLAS）在 float32 sgemm 大矩阵上会
  触发访问冲突崩溃；最近邻批量 Tanimoto 使用 numpy≥2.0 的 `np.bitwise_count`
  在 packbits 字节上做整数 AND + popcount，数值与浮点定义等价，因此 requirements
  要求 **numpy ≥ 2.0**。
- 最近邻 PDB ID 逐行输出在结果表的 `domain_nearest_train_pdb` 列，便于溯源核查。

## 8. 已知局限（如实披露）

1. **随机划分乐观偏差**：正式权重采用随机 KFold；骨架审计显示存在约 0.057 的
   R² 落差（见 §3）。排序参考可用，不宜当绝对结合能解读。
2. **化学空间覆盖稀疏区**：分子量 700–1300 的多环糖苷类大分子在 PDBBind 中覆盖
   稀疏，模型对该区域外推不确定性大；此类候选不应以模型分为唯一依据，建议结合
   分子对接、分子动力学等下游证据综合裁决。
3. **2D+快筛口径差异**：`predict.py`/`screen.py` 与全特征训练分布存在口径差异
   （见 §6.2 说明），输出宜作相对排序参考。
4. **蛋白/口袋残基特征**：训练集中 `prot_seq_len`/`prot_n_residues`/
   `pocket_n_residues` 是各复合物的真实残基数；推理时请尽量提供真实值
   （输入列或命令行参数），填 0 会带来分布偏移（脚本会警告）。

## 9. 引用

- PDBBind 数据库：Wang R., Fang X., Lu Y., Wang S. *The PDBbind Database:
  Collection of binding affinities for protein–ligand complexes with known
  three-dimensional structures.* J. Med. Chem. 2004, 47, 2977–2980；
  Liu Z. et al. *Forging the basis for database-level bioactivity prediction.*
  Nucleic Acids Res. 2015, 43(D1), D543–D551（数据使用请遵循其官方许可）。
- XGBoost：Chen T., Guestrin C. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- 适用域思想参见 QSAR 领域通行做法（最近邻相似度 AD 判据）。

## 10. 免责声明

本模型输出为**计算预测值**，仅用于候选分子排序辅助与优先级参考，不构成任何成药
结论或生物学保证；BORDERLINE / OUT_OF_DOMAIN 分子的预测与区间仅供参考，最终候选
应结合分子对接等下游计算与实验证据综合裁决。
