# config.py —— 路径、超参数、全局配置
# 模块：XGBoost 蛋白-配体结合强度（pKd）预测
# 训练数据：PDBBind v2020（获取方式见上级目录 README.md 数据来源节）

from pathlib import Path
import os

# ============================================================
# 1. 路径配置（后续改路径只改这里）
# ============================================================
# 数据根目录：PDBBind v2020.R1 解压位置（环境变量 PDBBIND_ROOT）
DATA_ROOT = Path(os.environ.get("PDBBIND_ROOT", ""))

# 蛋白-配体复合物的结构目录（解压后实际结构：P-L/<年代>/<pdbid>/）
#   例如：P-L/1981-2000/1a1e/{1a1e_protein.pdb, 1a1e_ligand.mol2, 1a1e_pocket.pdb}
COMPLEXES_DIR = DATA_ROOT / "P-L"

# 按年代划分的子目录（PDBBind v2020.R1 演示版分三组）
COMPLEXES_ERA_SUBDIRS = ["1981-2000", "2001-2010", "2011-2019"]

# 每个复合物目录的文件名格式：<pdbid>_protein.pdb / _ligand.mol2 / _ligand.sdf / _pocket.pdb
# 见 step1_load_data.py 中 resolve_complex_paths(pdbid) 函数
COMPLEX_FILE_TEMPLATES = {
    "protein": "{pdb}_protein.pdb",
    "ligand_mol2": "{pdb}_ligand.mol2",
    "ligand_sdf":  "{pdb}_ligand.sdf",
    "pocket":      "{pdb}_pocket.pdb",
}

# INDEX 文件（亲和力标签 + 元信息）
# index.tar.gz 实际解压到 DATA_ROOT/index/ 子目录，根下没有独立文件
INDEX_PL_FILE = DATA_ROOT / "index" / "INDEX_general_PL.2020R1.lst"

# 说明：v2020.R1 演示版未提供 refined set / core set (CASF-2016) 的显式区分
#      （索引只有 general_PL，无 refined/core 单独 INDEX）
#      应对方案 B：将全部 19,037 条视为 general set，
#                  1) 以分辨率 <= 2.5Å 且无 "incomplete/covalent/redundant" 备注的高质量样本
#                     自构造 "伪 refined set" 做训练
#                  2) 按 30% 蛋白序列相似度聚类 + 固定种子划分 train/val/test
#                  3) 后续如拿到 CASF-2016 core set 285 清单则替换为官方测试集
REFINED_RESOLUTION_CUTOFF = 2.5          # Å，分辨率优于此值才纳入伪refined
REFINED_DROP_REMARK_KEYWORDS = [
    "incomplete",     # 结构不完整
    "covalent",       # 共价复合物（结合机制不同）
    "redundant",      # 冗余样本
    "isomer",         # 同分异构体歧义
]


# 代码目录
CODE_ROOT = Path(__file__).resolve().parent

# 产出目录（训练产物统一落在 code/runs/ 下，不污染仓库其余部分）
OUTPUT_ROOT = CODE_ROOT / "runs" / "xgboost_train"
FEATURES_DIR = OUTPUT_ROOT / "features"
MODELS_DIR   = OUTPUT_ROOT / "models"
REPORTS_DIR  = OUTPUT_ROOT / "reports"

# 确保产出目录存在（安全，创建不出错即可）
for d in [FEATURES_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. PDBBind 标签解析参数
# ============================================================
# INDEX 第4列正则：匹配 "Ki=49uM" / "Kd<10nM" / "IC50<=3.5pM" 这类
# 提取 (类型, 比较符, 数值, 单位) 四元组
LABEL_PATTERN = r"^(Kd|Ki|IC50)\s*([<>=]*)\s*([\d.]+)\s*(fM|pM|nM|uM|mM|M)\s*$"

# 单位 -> M 换算系数
UNIT_TO_M = {
    "fM": 1e-15,
    "pM": 1e-12,
    "nM": 1e-9,
    "uM": 1e-6,
    "mM": 1e-3,
    "M":  1.0,
}

# 对不等式样本的处理策略：
#   "drop"   —— 删除 < / <= / > / >= 样本（最保守，避免泄漏）
#   "use"    —— 当精确值用（丢弃不等式标记，适合小数据）
#   默认 "use"（PDBBind 里不等式样本占比不高，且方向一致不会严重偏置）
INEQUALITY_POLICY = "use"

# ============================================================
# 3. 数据划分策略
# ============================================================
# 当前策略：随机固定种子划分 train/val/test；正式版本可替换为
# 30% 序列相似度聚类划分（mmseqs2/blastclust），避免同源泄漏。
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# ============================================================
# 4. 特征工程开关
# ============================================================
# Baseline 特征（必做）：1D/2D 描述符
FEAT_LIGAND_1D2D = True       # RDKit 描述符：分子量、LogP、TPSA、HBD/HBA 等
FEAT_LIGAND_MORGAN = True      # Morgan 指纹（半径 2，2048 位）
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# 优化特征 1：3D 相互作用（核心优化）—— PLIP 提取
FEAT_PLIP_INTERACTIONS = True  # 氢键/疏水/π-π/盐桥/接触残基 计数 + 原子对距离分箱
PLIP_MAXTHREADS = 4            # PLIP CLI 并行数（机器 CPU 核数一半较稳妥）

# 优化特征 2：Vina 能量项（默认不做，占位）
FEAT_VINA_ENERGY = False       # ⏸️ 需要批量 Vina 对接，先不启用

# 蛋白描述符（简化）：序列长度、口袋残基数
FEAT_PROTEIN_DESC = True

# ============================================================
# 5. XGBoost 训练超参数（Baseline 用默认，step4 用 Optuna 调参）
# ============================================================
XGB_BASE_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,              # 用满 CPU 核
    "objective": "reg:squarederror",
    "tree_method": "hist",     # 大数据快，xgboost >=1.3 支持
}

# 5 折交叉验证
N_SPLITS_CV = 5

# Optuna 调参搜索空间（step4_optimized_v3.py 用）
# 优化记录：50→20，配合 step4 搜索空间收紧 + gap 惩罚，目标压缩过拟合
OPTUNA_N_TRIALS = 20

# ============================================================
# 6. 评估指标阈值（用于打印合格/不合格）
# ============================================================
METRIC_THRESHOLDS = {
    "cv_r2_excellent": 0.7,
    "cv_r2_good":      0.6,     # 经验合格线
    "cv_r2_min":       0.4,     # < 0.4 视为失败，触发退路
    "cv_rmse_good":    1.4,     # log 单位
    "cv_rmse_max":     1.8,
}

# ============================================================
# 7. 应用阶段（可选）：上游检索/对接工具产出输入
# ============================================================
# 如需把本模型接到"检索工具 Top-N 分子"之后做复核打分，可设置：
#   形如 TOPN_POSES_DIR / <pdb_id或lig_id> / *_docked.pdb
TOPN_POSES_DIR = Path(os.environ.get("TOPN_POSES_DIR", ""))
TOPN_SMILES_CSV = Path(os.environ.get("TOPN_SMILES_CSV", ""))

# 靶点蛋白受体 PDB 路径（做应用阶段 3D/PLIP 特征提取时需要）
TARGET_PROTEIN_PDB = Path(os.environ.get("TARGET_PROTEIN_PDB", ""))
