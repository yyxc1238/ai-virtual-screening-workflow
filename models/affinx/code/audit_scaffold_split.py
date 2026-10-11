# -*- coding: utf-8 -*-
"""
audit_scaffold_split.py — v3 结合强度模型的 scaffold 泄露审计（补充评估，非重训正式模型）

动机
----
v3 正式模型的交叉验证采用随机 KFold（同骨架类似物可能同时落入训练/验证折），
CV R²=0.5629。随机划分会让"近邻化合物"互相抬分，存在乐观偏差。
本脚本用 Bemis-Murcko 骨架做 GroupKFold（同一骨架所有样本强制在同一折），
在**完全相同的样本、特征、v3 冻结超参**下与随机 KFold 对照，量化泄露幅度。

数据需求（不随包分发）
----------------------
1. features CSV：step1/step2 产物（默认 features_3d_vina_5316.csv），含 pdb,pKd 与特征列；
2. PDBBind v2020.R1 配体文件：$PDBBIND_ROOT/P-L/<era>/<pdb>/<pdb>_ligand.sdf
   （设置环境变量 PDBBIND_ROOT，或用 --pdbbind-root）；
3. CASF-2016 285 清单：与正式训练一致排除。

输出（--output-dir）
--------------------
- scaffold_map_<n>.csv         每个 pdb 的 Murcko 骨架（含缓存，重跑直接读）
- scaffold_audit_folds.csv     两种划分每折 R²/RMSE/MAE
- scaffold_audit_summary.txt   汇总结论

用法
----
    set PDBBIND_ROOT=<PDBBind 解压根目录>
    python audit_scaffold_split.py \
        --features <path>/features_3d_vina_5316.csv \
        --model-pkl ../xgboost_optimized_v3.pkl \
        --output-dir audit_out
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")  # 抑制批量解析时的坐标/价态刷屏告警（失败以 None 计）
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

ERAS = ["1981-2000", "2001-2010", "2011-2019"]
NO_SCAFFOLD = "_no_murcko_scaffold_"  # 无环/无骨架分子统一组（DeepChem 惯例）


def find_ligand_sdf(pdb_root: Path, pdb: str):
    for era in ERAS:
        fp = pdb_root / "P-L" / era / pdb / f"{pdb}_ligand.sdf"
        if fp.exists():
            return fp
    return None


def scaffold_of_sdf(fp: Path):
    # RDKit C++ fopen 在含非 ASCII 字符的路径上会失败，故先读文本再走 MolBlock
    try:
        block = fp.read_text(encoding="utf-8", errors="replace")
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True)
        if mol is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(sc)
        return smi if smi else NO_SCAFFOLD
    except Exception:
        return None


def build_scaffold_map(unique_pdbs, pdb_root: Path, cache_csv: Path):
    if cache_csv.exists():
        m = pd.read_csv(cache_csv, dtype={"pdb": str, "scaffold": str})
        done = set(m["pdb"])
        rows = m.to_dict("records")
    else:
        done, rows = set(), []
    todo = [p for p in unique_pdbs if p not in done]
    print(f"[scaffold] 缓存 {len(done)}，待解析 {len(todo)}")
    t0 = time.time()
    for i, pdb in enumerate(todo, 1):
        fp = find_ligand_sdf(pdb_root, pdb)
        smi = scaffold_of_sdf(fp) if fp else None
        rows.append({"pdb": pdb, "scaffold": smi if smi else "_PARSE_FAIL_"})
        if i % 500 == 0:
            pd.DataFrame(rows).to_csv(cache_csv, index=False)
            print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(cache_csv, index=False)
    return out


def cv_eval(X, y, splitter, groups=None):
    r2s, rmses, maes, sizes = [], [], [], []
    for tr, va in splitter.split(X, y, groups) if groups is not None else splitter.split(X):
        m = XGBRegressor(**PARAMS)
        m.fit(X[tr], y[tr], verbose=False)
        pred = m.predict(X[va])
        r2s.append(r2_score(y[va], pred))
        rmses.append(np.sqrt(mean_squared_error(y[va], pred)))
        maes.append(mean_absolute_error(y[va], pred))
        sizes.append(len(va))
    return np.array(r2s), np.array(rmses), np.array(maes), np.array(sizes)


PARAMS = {}  # 从 pkl 填充


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model-pkl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pdbbind-root", default=os.environ.get("PDBBIND_ROOT", ""))
    ap.add_argument("--casf-list", default="")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_root = Path(args.pdbbind_root)
    if not pdb_root.exists():
        sys.exit(f"[ERROR] PDBBIND_ROOT 不存在: {pdb_root}")

    pkg = pickle.load(open(args.model_pkl, "rb"))
    feature_cols = pkg["feature_cols"]
    PARAMS.update(pkg["params"])
    print(f"[audit] v3 超参载入: n_estimators={PARAMS.get('n_estimators')}, "
          f"max_depth={PARAMS.get('max_depth')}")

    df = pd.read_csv(args.features)
    print(f"[audit] 特征表 {len(df)} 行")
    casf = set()
    if args.casf_list and Path(args.casf_list).exists():
        c = pd.read_csv(args.casf_list, sep="\t")
        col = "#code" if "#code" in c.columns else c.columns[0]
        casf = set(c[col].str.lower())
        df = df[~df["pdb"].str.lower().isin(casf)].reset_index(drop=True)
        print(f"[audit] 排除 CASF {len(casf)} 后训练池 {len(df)}")

    cache_csv = out_dir / f"scaffold_map_{df['pdb'].nunique()}.csv"
    smap = build_scaffold_map(df["pdb"].drop_duplicates().tolist(), pdb_root, cache_csv)
    df = df.merge(smap, on="pdb", how="left")
    n_fail = int((df["scaffold"] == "_PARSE_FAIL_").sum())
    df = df[df["scaffold"] != "_PARSE_FAIL_"].reset_index(drop=True)
    print(f"[audit] 可评估 {len(df)}；解析失败剔除 {n_fail}")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    y = df["pKd"].values.astype(np.float32)
    groups = pd.factorize(df["scaffold"])[0]
    n_groups = len(set(groups))
    print(f"[audit] Murcko 组数 {n_groups}（含统一无骨架组）")

    fold_rows = []
    print("[audit] 随机 KFold（v3 同口径）...")
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    r2, rmse, mae, sizes = cv_eval(X, y, kf)
    for i in range(args.n_splits):
        fold_rows.append(dict(split="random_KFold", fold=i + 1, n_val=int(sizes[i]),
                              r2=r2[i], rmse=rmse[i], mae=mae[i]))
    print("[audit] Murcko GroupKFold（骨架不混折）...")
    gkf = GroupKFold(n_splits=args.n_splits)
    r2g, rmseg, maeg, sizesg = cv_eval(X, y, gkf, groups=groups)
    for i in range(args.n_splits):
        fold_rows.append(dict(split="murcko_GroupKFold", fold=i + 1, n_val=int(sizesg[i]),
                              r2=r2g[i], rmse=rmseg[i], mae=maeg[i]))
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out_dir / "scaffold_audit_folds.csv", index=False)

    def stat(a):
        return f"{a.mean():.4f} +/- {a.std(ddof=1):.4f}"

    top_sc = df["scaffold"].value_counts().head(10)
    with open(out_dir / "scaffold_audit_summary.txt", "w", encoding="utf-8") as f:
        f.write("XGBoost v3 scaffold split leakage audit\n")
        f.write("=" * 60 + "\n")
        f.write(f"samples (CASF excluded): {len(y)}\n")
        f.write(f"unique Murcko scaffolds: {n_groups}\n")
        f.write(f"samples/group mean: {len(y)/n_groups:.2f}\n")
        f.write(f"parse-failed dropped: {n_fail}\n\n")
        f.write(f"random  KFold R2  : {stat(r2)}   RMSE {stat(rmse)}   MAE {stat(mae)}\n")
        f.write(f"Murcko  GroupKF R2: {stat(r2g)}   RMSE {stat(rmseg)}   MAE {stat(maeg)}\n")
        f.write(f"R2 drop (random - scaffold): {r2.mean()-r2g.mean():.4f}\n\n")
        f.write("interpretation:\n")
        f.write("  drop <= 0.03  negligible random-split optimism;\n")
        f.write("  0.03-0.08     moderate, disclose and prefer scaffold splits in next model;\n")
        f.write("  > 0.08         material leakage, scaffold-split retraining required.\n\n")
        f.write("top 10 scaffold groups (sample counts):\n")
        for smi, n in top_sc.items():
            f.write(f"  {n:5d}  {smi[:80]}\n")
    print(folds.to_string(index=False))
    print(f"[audit] R2: random {r2.mean():.4f} | scaffold {r2g.mean():.4f} "
          f"| drop {r2.mean()-r2g.mean():.4f}")
    print(f"[audit] 报告: {out_dir/'scaffold_audit_summary.txt'}")


if __name__ == "__main__":
    main()
