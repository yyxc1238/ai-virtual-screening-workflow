# 主干计算管线脚本

> 与 `README.md` 方法文档（00–10、12 篇）一一对应的可执行管线。每个脚本都是**通用工具**：
> 靶点、分子、路径全部走命令行参数，不绑定任何特定项目。阈值均为 CLI 参数 +
> 文档口径默认值，**应在看到结果之前确定并冻结**（公平性与可复现性的前提）。

---

## 运行顺序总览

```
01 结构预处理 -> 02 口袋识别/盒子 -> 03 协议验证 -> 04 库清洗分轨
    -> 05 批量对接(第一门) -> 06 物理规则质检 -> 07 秩融合/可开发性提醒 -> 08 MD 终裁分析
```

| 脚本 | 功能 | 对应方法文档 | 主要依赖 |
|---|---|---|---|
| `01_prepare_structure.py` | PDBFixer 补缺失/加氢/去杂 + AlphaFold pLDDT 质量报告 | 02 篇 | pdbfixer, openmm |
| `02_identify_pockets.py` | fpocket 口袋扫描 + 参考配体锚点筛选 + 对接盒定义 | 03 篇 | fpocket（外部命令）, numpy |
| `03_validate_protocol.py` | 筛选前协议验证：redocking（RMSD<=2Å）/ 阳性锚点 | 05 篇 2.6/3.0 | vina, meeko, rdkit |
| `04_prepare_library.py` | 库清洗（去盐/标准化/去重/全氢校验）+ 重原子数分轨 | 04 篇 | rdkit |
| `05_dock_screen.py` | Meeko+Vina 批量对接（多盒子）+ 第一门名单 | 05 篇 | vina, meeko, rdkit |
| `06_posebusters_qc.py` | PoseBusters 物理规则质检 + 失败类型分布 | 05 篇 2.3 | posebusters, rdkit |
| `07_fusion_rank.py` | 秩归一化加权融合排名 + MW/LogP 可开发性提醒 | 08 篇 | pandas, rdkit |
| `08_md_analysis.py` | MD 轨迹 B1–B4 机械裁决，逐 pose verdict JSON | 09 篇 2.5 | MDAnalysis, numpy |

各脚本的详细参数说明用 `python <脚本> --help` 查看（均为中文帮助）。

---

## 依赖安装

```bash
# 1) Python 依赖（rdkit 建议走 conda-forge，其余走 pip）
pip install -r requirements.txt
#    或 conda install -c conda-forge rdkit numpy pandas

# 2) 结构预处理（01）推荐 conda 安装 pdbfixer + openmm
conda install -c conda-forge pdbfixer openmm

# 3) 外部命令行工具
#    fpocket（02）：conda install -c conda-forge fpocket  或 apt install fpocket
#    Vina 通过 vina Python 包调用（03/05），无需单独二进制；
#    受体 PDBQT 需提前制备（如 ADFR 套件 prepare_receptor、OpenBabel）。
```

每个脚本启动时都会检查依赖，缺失时给出对应的安装提示。

---

## 快速上手（通用占位符示例）

```bash
# 1. 结构预处理（含预测结构 pLDDT 报告）
python 01_prepare_structure.py --input receptor_raw.pdb --output receptor_clean.pdb
python 01_prepare_structure.py --input af_model.pdb --plddt-only

# 2. 口袋识别与对接盒定义（有共晶配体时加 --ref-ligand 做锚点交叉验证）
python 02_identify_pockets.py --receptor receptor_clean.pdb --ref-ligand ligand.pdb
#    读取输出的 box center/size，作为 03/05 的 --center/--size

# 3. 筛选前协议验证（文档：验证不过，不许进入全库筛选）
python 03_validate_protocol.py --mode redock --receptor receptor.pdbqt \
    --ligand lig_dock.pdbqt --ref lig_crystal.pdb --center X Y Z --size SX SY SZ
python 03_validate_protocol.py --mode anchor --receptor receptor.pdbqt \
    --library known_actives.smi --anchor-name anchor01 --center X Y Z --size SX SY SZ

# 4. 分子库清洗与分轨（轻线/重线）
python 04_prepare_library.py --input library.smi --out-dir prepared_library

# 5. 批量对接 + 第一门（--box 可多次指定，多口袋并行）
python 05_dock_screen.py --receptor receptor.pdbqt --input prepared_library/light_track.sdf \
    --box CX CY CZ SX SY SZ --out-dir docking_out

# 6. 构象物理规则质检（05 用 --write-poses 输出的 pose，或 vina 输出转 SDF）
python 06_posebusters_qc.py --poses poses.sdf --receptor receptor.pdb --out-dir qc_out

# 7. 多证据秩融合（对接分属"越小越好"，其余按实际方向声明；权重先冻结再出结果）
python 07_fusion_rank.py --input scores.csv --scores vina,affinity_model \
    --lower-is-better vina --weights 0.5,0.5 --smiles-col smiles --out fusion_rank.csv

# 8. MD 终裁分析（每条轨迹 = 一个 pose 的短 MD；任一 pose 全 PASS 即分子存活）
python 08_md_analysis.py --molecule cand01 --topol md.tpr --traj pose1.xtc --traj pose2.xtc \
    --ligand-sel "resname LIG and not name H*" --site-sel "protein and around 6 of resname LIG"
```

---

## 使用要点（均来自方法文档，跑之前务必读对应篇章）

1. **先验证后开跑**（05 篇 3.0）：03 的验证报告未 PASS 前，不要把算力投到全库上。
2. **公平性铁律**（05 篇 2.4）：同一条筛选线上所有分子用完全相同的引擎、盒子、采样参数；
   异常正分标记处理，不当作低分参与排序；不同引擎分数不混排。
3. **物理门**（05 篇 2.3）：06 淘汰率异常高（>25%）时先排查加氢/质子化/格式，而非调阈值。
4. **融合**（08 篇）：权重在看到结果之前冻结；模型分只投票、不设硬门槛；
   短名单还应叠加完整的可开发性四维评分与骨架多样性约束（人工/外部工具完成）。
5. **MD 终裁**（09 篇）：B1–B4 阈值在看结果前冻结；"任一 pose 全 PASS 即存活"；
   多轮结论并存留档；未部署 MD 的分子显式标注"未部署"。
6. **数字口径**（04 篇）：区分"原始下载数 / 清洗后入库数 / 进入正式筛选数"，
   各脚本的台账（log/summary/JSON）即为对账依据。

## 方法论依据

本目录脚本只实现流程与阈值口径，方法原理、适用边界与常见陷阱
请阅读仓库根目录对应编号的方法文档（01–10、12 篇）。
