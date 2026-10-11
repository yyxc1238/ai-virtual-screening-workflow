# -*- coding: utf-8 -*-
"""
screen.py —— 小分子库一键虚拟快筛入口（纯 CPU）
=====================================================================
定位：
  对一个待筛分子库批量执行  AffinX（XGBoost v3）打分 → 训练适用域判定 →
  pKd 排序输出 TopK。本入口只依赖 RDKit + XGBoost + numpy/pandas，
  不需要分子对接/动力学等外部工具，普通笔记本即可运行。

口径说明：
  1) 本脚本只复现 XGBoost 打分与适用域部分，用于海量库的快速优先级排序；
     正式筛选还可在其后串联分子对接、成药性门控与动力学验证等环节
     （这些编排属于管线，不属于本模型仓库）。
  2) 本入口无 Vina 能量，vina_total 特征按 0 输入（Δ-learning 口径下
     相当于"不给 3D 先验"），得到的是纯 2D 快筛 pKd。
  3) OUT_OF_DOMAIN / BORDERLINE 分子的 pKd 为外推/半外推结果，
     ad_recommendation 列给出处置建议，排序时不得与域内分子同等信任。
  4) 蛋白/口袋残基数特征通过 --prot_seq_len/--prot_n_residues/
     --pocket_n_residues 传入（见 predict.py 同名说明）。

输入格式（自动识别）：
  .csv  —— SMILES 列名兼容 smiles/SMILES/canonical_smiles；
            ID 列名兼容 mol_id/ID/id/name（缺失则自动编号 s001…）；
            其余列原样保留在输出中。
  .smi  —— 每行 "SMILES [ID]"，# 开头或空行跳过。

用法：
  python screen.py --input mylib.csv --topk 50
  python screen.py --input mylib.smi --output run1.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict import (load_model, compute_2d_features)
from applicability_domain import (FpReference, morgan_bits,
                                  domain_flag, uncertainty_sigma)

MODEL_VERSION = "xgboost_optimized_v3"

SMILES_CANDIDATES = ["smiles", "SMILES", "canonical_smiles", "smi"]
ID_CANDIDATES = ["mol_id", "design_id", "ID", "id", "name", "molecule_id"]

AD_ADVICE = {
    "IN_DOMAIN": "域内预测：可优先推进下游验证（对接/成药性门控等）",
    "BORDERLINE": "边界域：可进入下游验证，但 pKd 需谨慎解读、以其他证据为准",
    "OUT_OF_DOMAIN": "域外预测：pKd 属外推，仅建议作化学多样性探索，不得直接采信",
}


def load_library(path: Path) -> pd.DataFrame:
    """读 .csv/.smi 分子库，规范成 mol_id/smiles 两列（+ 附带列）。"""
    if path.suffix.lower() == ".smi":
        ids, smiles = [], []
        for k, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            s = parts[0]
            mid = parts[1] if len(parts) > 1 else f"s{k:03d}"
            smiles.append(s)
            ids.append(mid)
        return pd.DataFrame({"mol_id": ids, "smiles": smiles})

    df = pd.read_csv(path, dtype=str)
    scol = next((c for c in SMILES_CANDIDATES if c in df.columns), None)
    if scol is None:
        sys.exit(f"[ERROR] {path.name} 中找不到 SMILES 列（尝试过 {SMILES_CANDIDATES}）")
    idcol = next((c for c in ID_CANDIDATES if c in df.columns), None)
    out = pd.DataFrame()
    out["mol_id"] = df[idcol] if idcol else [f"s{i:03d}" for i in range(1, len(df) + 1)]
    out["smiles"] = df[scol].astype(str).str.strip()
    # 保留来源等附加信息（不与输出保留列重名的列）
    for c in df.columns:
        if c not in (scol, idcol) and c not in out.columns:
            out[c] = df[c]
    return out


def simple_descriptors(smiles: str):
    """轻量 triage 描述符（RDKit 失败返回 None）。"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {"MW": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 2)}


def main():
    ap = argparse.ArgumentParser(description="小分子库 XGBoost 快筛（含适用域）")
    ap.add_argument("--input", required=True,
                    help="分子库 .csv（含 SMILES 列）或 .smi")
    ap.add_argument("--output", default="screen_ranked.csv",
                    help="排序后输出 CSV 路径")
    ap.add_argument("--topk", type=int, default=30,
                    help="只保留 pKd 排名前 K（默认 30；全部保留传 0 或负数）")
    ap.add_argument("--min-tanimoto", type=float, default=0.0,
                    help="可选：先按最近邻 Tanimoto 阈值过滤再排序（默认 0=不过滤）")
    ap.add_argument("--prot_seq_len", type=float, default=None,
                    help="靶点蛋白序列长度（默认填 0，见 predict.py 说明）")
    ap.add_argument("--prot_n_residues", type=float, default=None,
                    help="靶点蛋白残基数")
    ap.add_argument("--pocket_n_residues", type=float, default=None,
                    help="口袋残基数")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[ERROR] 分子库不存在: {in_path}")

    lib = load_library(in_path)
    lib = lib[lib["smiles"].str.len() > 0].reset_index(drop=True)
    print(f"[screen] 读入 {len(lib)} 个分子 <- {in_path.name}")

    model, feature_cols = load_model()
    print(f"[screen] 模型 {MODEL_VERSION} | 特征维度 {len(feature_cols)}")
    fp_ref = FpReference()
    print(f"[screen] 适用域参考库 {len(fp_ref.ids)} 条训练池指纹")

    prot = {"prot_seq_len": args.prot_seq_len or 0.0,
            "prot_n_residues": args.prot_n_residues or 0.0,
            "pocket_n_residues": args.pocket_n_residues or 0.0}
    if prot["prot_n_residues"] == 0 or prot["pocket_n_residues"] == 0:
        print("[screen] [WARN] 蛋白/口袋残基特征未提供，按 0 填充（分布偏移，"
              "请用 --prot_seq_len/--prot_n_residues/--pocket_n_residues 提供真实值）")

    rows, fp_list, invalid = [], [], 0
    for _, r in lib.iterrows():
        smiles = r["smiles"]
        feats = compute_2d_features(smiles)
        qbits = morgan_bits(smiles)
        if feats is None or qbits is None:
            invalid += 1
            continue
        feats.update(prot)
        feats["vina_total"] = 0.0          # 快筛无 3D 先验
        rows.append({"mol_id": r["mol_id"], "smiles": smiles,
                     "feats": feats,
                     **{c: r[c] for c in lib.columns
                        if c not in ("mol_id", "smiles") and pd.notna(r.get(c))}})
        fp_list.append(qbits)
    print(f"[screen] SMILES 有效 {len(rows)} / 无效 {invalid}")

    # --- 批量 XGBoost 打分 ---
    X = np.array([[row["feats"].get(c, 0.0) for c in feature_cols]
                  for row in rows], dtype=np.float32)
    pkd = model.predict(X).astype(float)

    # --- 批量适用域最近邻（packbits 字节 AND+popcount，绕开 BLAS）---
    sim, nn_idx = fp_ref.nearest_neighbor_batch(np.asarray(fp_list, dtype=np.uint8))

    for i, row in enumerate(rows):
        s = float(sim[i])
        flag = domain_flag(s)
        sigma = uncertainty_sigma(s)
        p = float(pkd[i])
        row.pop("feats")
        desc = simple_descriptors(row["smiles"]) or {}
        row.update({
            "MW": desc.get("MW", ""),
            "LogP": desc.get("LogP", ""),
            "TPSA": desc.get("TPSA", ""),
            "xgb_pKd_v3": round(p, 4),
            "xgb_IC50_uM_v3": round((10 ** (-p)) * 1e6, 2),
            "domain_flag": flag,
            "domain_max_tanimoto": round(s, 4),
            "domain_nearest_train_pdb": str(fp_ref.ids[int(nn_idx[i])]),
            "pKd_ci95_low": round(p - 1.96 * sigma, 3),
            "pKd_ci95_high": round(p + 1.96 * sigma, 3),
            "ad_recommendation": AD_ADVICE[flag],
        })

    out = pd.DataFrame(rows)
    if args.min_tanimoto and args.min_tanimoto > 0:
        before = len(out)
        out = out[out["domain_max_tanimoto"] >= args.min_tanimoto].reset_index(drop=True)
        print(f"[screen] Tanimoto≥{args.min_tanimoto} 过滤: {before} -> {len(out)}")

    # 排序：域内优先（同档内按 pKd 降序），避免高分域外分子误导优先级
    flag_order = {"IN_DOMAIN": 0, "BORDERLINE": 1, "OUT_OF_DOMAIN": 2}
    out["_ad_order"] = out["domain_flag"].map(flag_order)
    out = out.sort_values(["_ad_order", "xgb_pKd_v3"],
                          ascending=[True, False]).reset_index(drop=True)
    out = out.drop(columns="_ad_order")
    if args.topk and args.topk > 0:
        out = out.head(args.topk).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[screen] 输出 Top{len(out)} -> {out_path}")
    show = out[["rank", "mol_id", "xgb_pKd_v3", "domain_flag",
                "domain_max_tanimoto"]].head(min(10, len(out)))
    print(show.to_string(index=False))
    print("[screen] 提示：本结果为纯 2D 快筛优先级；正式候选须再过分子对接、"
          "成药性门控与动力学验证等下游环节。")


if __name__ == "__main__":
    main()
