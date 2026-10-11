# step1_load_data.py —— 数据加载 + 标签统一 + 质量筛选 + 数据划分
# 输入：PDBBind v2020.R1 INDEX + 复合物结构目录
# 输出：DATASET.csv（PDB码、元信息、pKd标签、结构路径），保存在 FEATURES_DIR
#
# 执行流程：
#   1. 解析 INDEX_general_PL.2020R1.lst → DataFrame[ pdb, resolution, year, bind_raw, affinity_type, value, unit, inequality, pKd, remark ]
#   2. 质量筛选（伪refined set）：分辨率<=2.5Å 且 备注不含 incomplete/covalent/redundant/isomer
#   3. 解析复合物结构路径：遍历 COMPLEXES_ERA_SUBDIRS 找到每个 pdb 的目录
#   4. 数据划分：固定种子分 train/val/test（序列相似度聚类防同源泄漏放到后续版本）
#   5. 持久化输出：DATASET.csv（后续特征、训练步骤直接读此文件）

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 允许直接运行：python step1_load_data.py
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from config import (
    DATA_ROOT, COMPLEXES_DIR, COMPLEXES_ERA_SUBDIRS, COMPLEX_FILE_TEMPLATES,
    INDEX_PL_FILE, LABEL_PATTERN, UNIT_TO_M, INEQUALITY_POLICY,
    REFINED_RESOLUTION_CUTOFF, REFINED_DROP_REMARK_KEYWORDS,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    FEATURES_DIR,
)

# 输出文件
DATASET_CSV = FEATURES_DIR / "DATASET.csv"
DATASET_REFIND_CSV = FEATURES_DIR / "DATASET_refined.csv"


# ============================================================
# 1. INDEX 解析
# ============================================================
def parse_index_file(index_path: Path) -> pd.DataFrame:
    """解析 INDEX_general_PL.2020R1.lst，返回包含原始字段的 DataFrame。

    索引文件格式（注释行以 # 开头）：
        PDB码   分辨率   年份   Kd/Ki/IC50=值+单位   // 参考.pdf (配体名) 备注
    例：
        4tmn  1.70  1989  Ki=0.068nM    // 4tmn.pdf (0PK)
        4cts  2.90  1984  Kd<10uM       // 4cts.pdf (OAA)
        2mip  2.20  1993  IC50<=3.5nM   // 2mip.pdf (7-mer)
        6rsa  NMR   1986  Ki=40uM       // 6rsa.pdf (UVC)     ← 分辨率 NMR
    """
    rows = []
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            # 拆分数据字段(前4列) + 注释(//后面)
            if "//" in line:
                data_part, remark_part = line.split("//", 1)
            else:
                data_part, remark_part = line, ""
            tokens = data_part.split()
            if len(tokens) < 4:
                continue
            pdb, resolution, year, bind_raw = tokens[0], tokens[1], tokens[2], tokens[3]
            remark = remark_part.strip()
            rows.append({
                "pdb": pdb,
                "resolution_raw": resolution,   # 可能是 "2.10" 或 "NMR"
                "year": int(year),
                "bind_raw": bind_raw,
                "remark": remark,
            })
    return pd.DataFrame(rows)


def parse_binding_label(row: pd.Series) -> pd.Series:
    """解析 bind_raw，提取 (affinity_type, 比较符, 数值, 单位, pKd)。
    解析失败返回 NaN（会被后续 drop 去掉）。
    """
    bind_raw = row["bind_raw"]
    m = re.match(LABEL_PATTERN, bind_raw.strip())
    if not m:
        return pd.Series([np.nan] * 5, index=["affinity_type", "inequality", "value", "unit", "pKd"])
    atype, op, val, unit = m.group(1), m.group(2), float(m.group(3)), m.group(4)
    coeff = UNIT_TO_M.get(unit, np.nan)
    if np.isnan(coeff):
        return pd.Series([atype, op, val, unit, np.nan], index=["affinity_type", "inequality", "value", "unit", "pKd"])
    molar = val * coeff
    if molar <= 0:
        pkd = np.nan
    else:
        import math
        pkd = -math.log10(molar)
    # 不等式处理
    if op not in ("", "=", "==") and INEQUALITY_POLICY == "drop":
        pkd = np.nan
    return pd.Series([atype, op, val, unit, pkd], index=["affinity_type", "inequality", "value", "unit", "pKd"])


def resolution_to_float(raw: str) -> float:
    """字符串分辨率→float；"NMR" 视为缺失（用 NaN），后续筛选会丢 NMR 样本"""
    try:
        return float(raw)
    except (ValueError, TypeError):
        return np.nan


# ============================================================
# 2. 复合物结构路径解析
# ============================================================
def build_pdb_dir_index(complexes_dir: Path, era_subdirs):
    """扫描 COMPLEXES_DIR/<era>/<pdbid>/，构建 {pdbid: 目录Path} 索引。
    pdbid 小写匹配（PDBBind目录名小写）。"""
    mapping = {}
    for era in era_subdirs:
        era_dir = complexes_dir / era
        if not era_dir.exists():
            continue
        for sub in era_dir.iterdir():
            if sub.is_dir():
                mapping[sub.name.lower()] = sub
    return mapping


def resolve_complex_paths(pdb: str, pdb_dir: Path) -> dict | None:
    """返回 protein/ligand_mol2/ligand_sdf/pocket 的路径 dict；若关键文件缺失返回 None。"""
    out = {}
    ok = True
    for key, tmpl in COMPLEX_FILE_TEMPLATES.items():
        p = pdb_dir / tmpl.format(pdb=pdb.lower())
        out[f"path_{key}"] = str(p)
        if key in ("protein", "ligand_mol2"):  # 这两个是后续特征提取必需
            if not p.exists():
                ok = False
    if not ok:
        return None
    return out


# ============================================================
# 3. 质量筛选（构造伪refined set）
# ============================================================
def mask_pseudo_refined(df: pd.DataFrame) -> pd.Series:
    """返回布尔mask：True = 进入伪refined set（高质量训练候选）"""
    res_ok = (df["resolution"].notna()) & (df["resolution"] <= REFINED_RESOLUTION_CUTOFF)
    pkd_ok = df["pKd"].notna()
    remark_lower = df["remark"].str.lower()
    bad_remark = pd.Series(False, index=df.index)
    for kw in REFINED_DROP_REMARK_KEYWORDS:
        bad_remark = bad_remark | remark_lower.str.contains(kw, na=False)
    return res_ok & pkd_ok & (~bad_remark)


# ============================================================
# 4. 数据划分（简化占位，第一版随机种子划分；序列相似度聚类放到后续完整版）
# ============================================================
def split_train_val_test(df: pd.DataFrame):
    """按 RANDOM_SEED 划分 train/val/test。
    TODO[v2]：用 mmseqs2/blastclust 按 30% 蛋白序列相似度聚类，再做划分，避免同源泄漏。
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    splits = np.empty(n, dtype=object)
    splits[:] = "train"
    splits[n_train:n_train + n_val] = "val"
    splits[n_train + n_val:] = "test"
    # 写回原顺序
    out = np.empty(n, dtype=object)
    out[idx] = splits
    return out


# ============================================================
# main
# ============================================================
def main():
    print("[step1] 开始：加载 INDEX 文件 ...")
    df = parse_index_file(INDEX_PL_FILE)
    print(f"  - INDEX 原始条目: {len(df)}")

    # 分辨率转 float
    df["resolution"] = df["resolution_raw"].apply(resolution_to_float)

    # 亲和力标签解析（向量级 apply）
    labels = df.apply(parse_binding_label, axis=1)
    df = pd.concat([df, labels], axis=1)
    print(f"  - pKd 解析成功: {df['pKd'].notna().sum()} / {len(df)}")

    # 结构路径匹配
    print("[step1] 扫描复合物目录，匹配路径 ...")
    pdb_dir_idx = build_pdb_dir_index(COMPLEXES_DIR, COMPLEXES_ERA_SUBDIRS)
    print(f"  - 扫描到复合物目录数: {len(pdb_dir_idx)}")
    paths = []
    for pdb in df["pdb"].str.lower():
        d = pdb_dir_idx.get(pdb)
        if d is None:
            paths.append({})
        else:
            r = resolve_complex_paths(pdb, d)
            paths.append(r if r is not None else {})
    paths_df = pd.DataFrame(paths)
    df = pd.concat([df.reset_index(drop=True), paths_df], axis=1)
    has_structure = df["path_protein"].notna() & df["path_ligand_mol2"].notna()
    print(f"  - 找到结构文件: {has_structure.sum()} / {len(df)}")

    # 保存完整（全量 general set）数据集
    df.to_csv(DATASET_CSV, index=False, encoding="utf-8-sig")
    print(f"[step1] 已保存全量数据集: {DATASET_CSV} (行数: {len(df)})")

    # 构造伪 refined set
    mask = mask_pseudo_refined(df) & has_structure
    refined_df = df.loc[mask].copy()
    print(f"[step1] 伪refined set 筛选后: {len(refined_df)}")
    if len(refined_df) == 0:
        print("  ⚠️ 空集！请降低 REFINED_RESOLUTION_CUTOFF 或放宽 REMARK 过滤")
        return

    # 划分 train/val/test
    refined_df["split"] = split_train_val_test(refined_df)
    print(f"  - split: train={ (refined_df['split']=='train').sum() }, "
          f"val={ (refined_df['split']=='val').sum() }, test={ (refined_df['split']=='test').sum() }")
    refined_df.to_csv(DATASET_REFIND_CSV, index=False, encoding="utf-8-sig")
    print(f"[step1] 已保存伪refined数据集（带split）: {DATASET_REFIND_CSV}")
    print("  ✅ step1 完成。下游：step2_extract_features.py 读 DATASET_refined.csv 做特征提取")


if __name__ == "__main__":
    main()
