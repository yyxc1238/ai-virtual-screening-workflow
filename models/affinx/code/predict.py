# -*- coding: utf-8 -*-
"""
predict.py —— 蛋白-配体结合强度（pKd）单表预测入口
=====================================================================
功能：
  读取候选分子表（mol_id + SMILES），加载 AffinX（XGBoost v3 结合强度模型），
  输出逐分子预测表（pKd / IC50 / 适用域标记 / 95% 近似区间）。

模型：../xgboost_optimized_v3.pkl（Δ-learning；2110 特征 =
      18 个 2D 描述符 + 2048 bit Morgan 指纹(r=2) + 蛋白/口袋 3 维 + PLIP 3D 接触
      + 6 个 Vina 能量项；训练池 PDBBind v2020 共 13,770 条，详见上级目录 README.md）。

推理口径（"2D+快筛"）：
  2D 描述符 + Morgan 指纹由 RDKit 从 SMILES 实时计算；
  vina_total 取输入表 vina_docking_score 列（无则填 0）；
  其余 3D/PLIP 接触特征填 0。因此本入口给出的是"2D+快筛口径"的 pKd 预测；
  如需全特征口径，需另行准备对接特征列（原管线中由独立的特征合并脚本完成，
  该编排脚本未随本仓库发布）。

蛋白/口袋特征（prot_seq_len / prot_n_residues / pocket_n_residues）：
  训练集中这三个特征是各复合物的真实残基数。推理时按以下优先级取值：
    1) 输入 CSV 若含同名列，直接使用；
    2) 否则用命令行参数 --prot_seq_len / --prot_n_residues / --pocket_n_residues；
    3) 都没有则填 0 并给出警告（会带来分布偏移，请尽量提供真实残基数）。

预测不确定性与适用域：
  每条预测附 4 列——domain_max_tanimoto（查询分子与 13,770 条训练配体的
  Morgan 指纹最近邻 Tanimoto）、domain_nearest_train_pdb（最近邻 PDB ID）、
  domain_flag（IN_DOMAIN≥0.50 / BORDERLINE 0.30-0.50 / OUT_OF_DOMAIN<0.30）、
  pKd_ci95_low/high（以五折 CV RMSE=1.2318 pKd 为基准、按最近邻距离线性放大
  1-2 倍后的 ±1.96σ 近似区间）。方法细节见 applicability_domain.py 与 README.md。

用法：
  python predict.py --input my.csv --output results.csv \
      --prot_seq_len 400 --prot_n_residues 400 --pocket_n_residues 20
  输入表必需列：mol_id, smiles；可选列：vina_docking_score, note 及任意附加列
  （附加列原样保留在输出中）。
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from applicability_domain import FpReference, assess as ad_assess

MODEL_PATH = Path(__file__).resolve().parents[1] / "xgboost_optimized_v3.pkl"
MODEL_VERSION = "xgboost_optimized_v3"

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048


def compute_2d_features(smiles: str):
    """SMILES -> 18 个 desc_ + 2048 个 fp_morgan_（命名与训练特征一致）。"""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molH = Chem.AddHs(mol)
    feats = {}
    desc_fns = [
        ("MolWt", Descriptors.MolWt), ("MolLogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA), ("HBD", Descriptors.NumHDonors),
        ("HBA", Descriptors.NumHAcceptors), ("RotatableBonds", Descriptors.NumRotatableBonds),
        ("AromaticRings", Descriptors.NumAromaticRings), ("Rings", Descriptors.RingCount),
        ("HeavyAtoms", Descriptors.HeavyAtomCount), ("FractionCSP3", Descriptors.FractionCSP3),
        ("NumHeteroatoms", Descriptors.NumHeteroatoms), ("NHOHCount", Descriptors.NHOHCount),
        ("NOCount", Descriptors.NOCount), ("MolMR", Descriptors.MolMR),
        ("MaxPartialCharge", getattr(Descriptors, "MaxPartialCharge", lambda m: 0.0)),
        ("MinPartialCharge", getattr(Descriptors, "MinPartialCharge", lambda m: 0.0)),
        ("MaxAbsPartialCharge", getattr(Descriptors, "MaxAbsPartialCharge", lambda m: 0.0)),
        ("MinAbsPartialCharge", getattr(Descriptors, "MinAbsPartialCharge", lambda m: 0.0)),
    ]
    for name, fn in desc_fns:
        try:
            feats[f"desc_{name}"] = float(fn(molH))
        except Exception:
            feats[f"desc_{name}"] = 0.0
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)
    arr = np.zeros(MORGAN_NBITS, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(molH), arr)
    for i in range(MORGAN_NBITS):
        feats[f"fp_morgan_{i}"] = int(arr[i])
    return feats


def load_model():
    with open(MODEL_PATH, "rb") as fh:
        pkg = pickle.load(fh)
    return pkg["model"], pkg["feature_cols"]


def main():
    ap = argparse.ArgumentParser(description="候选小分子 pKd 一键预测（含适用域与置信区间）")
    ap.add_argument("--input", required=True,
                    help="候选表 CSV：必需 mol_id, smiles；可选 vina_docking_score, note 等")
    ap.add_argument("--output", default="results.csv", help="输出结果表路径")
    ap.add_argument("--prot_seq_len", type=float, default=None,
                    help="靶点蛋白序列长度（输入表无该列时生效）")
    ap.add_argument("--prot_n_residues", type=float, default=None,
                    help="靶点蛋白残基数（输入表无该列时生效）")
    ap.add_argument("--pocket_n_residues", type=float, default=None,
                    help="口袋残基数（输入表无该列时生效）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[ERROR] 输入文件不存在: {in_path}")

    df = pd.read_csv(in_path, dtype={"mol_id": str})
    required = {"mol_id", "smiles"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] 输入表缺少必需列: {sorted(missing)}")

    print(f"[predict] 输入 {len(df)} 个分子 <- {in_path.name}")
    model, feature_cols = load_model()
    print(f"[predict] 模型 {MODEL_VERSION} | 特征维度 {len(feature_cols)}")
    try:
        fp_ref = FpReference()
        print(f"[predict] 适用域参考库: {len(fp_ref.ids)} 条训练池指纹 -> "
              f"输出含 domain_flag 与 pKd 95% 近似区间")
    except FileNotFoundError as exc:
        print(f"[predict] [WARN] {exc}")
        fp_ref = None

    rows, fail = [], 0
    for _, r in df.iterrows():
        smiles = str(r["smiles"]).strip()
        feats = compute_2d_features(smiles)
        if feats is None:
            fail += 1
            print(f"  ! SMILES 解析失败: {r['mol_id']}")
            continue
        # 蛋白/口袋残基数：输入列 > CLI 参数 > 0（警告）
        seq_len = r.get("prot_seq_len", np.nan)
        n_res = r.get("prot_n_residues", np.nan)
        pk_res = r.get("pocket_n_residues", np.nan)
        feats["prot_seq_len"] = float(seq_len) if pd.notna(seq_len) else (
            args.prot_seq_len if args.prot_seq_len is not None else 0.0)
        feats["prot_n_residues"] = float(n_res) if pd.notna(n_res) else (
            args.prot_n_residues if args.prot_n_residues is not None else 0.0)
        feats["pocket_n_residues"] = float(pk_res) if pd.notna(pk_res) else (
            args.pocket_n_residues if args.pocket_n_residues is not None else 0.0)
        if (feats["prot_seq_len"] == 0 or feats["prot_n_residues"] == 0
                or feats["pocket_n_residues"] == 0):
            print("  [WARN] 蛋白/口袋残基特征缺失，按 0 填充（分布偏移，请用 "
                  "--prot_seq_len/--prot_n_residues/--pocket_n_residues 提供真实值）")
        vina = r.get("vina_docking_score", np.nan)
        feats["vina_total"] = float(vina) if pd.notna(vina) else 0.0
        X = np.array([[feats.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
        pkd = float(model.predict(X)[0])

        row = {
            "mol_id": str(r["mol_id"]),
            "smiles": smiles,
            "xgb_pKd_v3": round(pkd, 4),
            "xgb_IC50_uM_v3": round((10 ** (-pkd)) * 1e6, 2),
            "vina_kcal_mol": float(vina) if pd.notna(vina) else "",
            "domain_flag": "",
            "domain_max_tanimoto": "",
            "domain_nearest_train_pdb": "",
            "pKd_ci95_low": "",
            "pKd_ci95_high": "",
            "model_version": MODEL_VERSION,
            "note": str(r.get("note", "")) if pd.notna(r.get("note", "")) else "",
        }
        # 适用域 + 不确定性（±1.96σ 高斯残差近似区间，sigma 按最近邻距离放大）
        if fp_ref is not None:
            ad = ad_assess(smiles, fp_ref)
            if ad is not None:
                sigma = ad["sigma_pKd"]
                row["domain_flag"] = ad["domain_flag"]
                row["domain_max_tanimoto"] = ad["domain_max_tanimoto"]
                row["domain_nearest_train_pdb"] = ad["domain_nearest_train_pdb"]
                row["pKd_ci95_low"] = round(pkd - 1.96 * sigma, 3)
                row["pKd_ci95_high"] = round(pkd + 1.96 * sigma, 3)
        # 附加输入列原样保留
        for c in df.columns:
            if c not in ("mol_id", "smiles", "vina_docking_score", "note") and c not in row:
                row[c] = r[c]
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("xgb_pKd_v3", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[predict] 成功 {len(out)} / 失败 {fail} -> {out_path}")
    print(out[["rank", "mol_id", "xgb_pKd_v3",
               "domain_flag", "domain_max_tanimoto"]].head(20).to_string(index=False))
    print("\n[predict] 完成。说明：本输出为 2D+快筛口径模型分；OUT_OF_DOMAIN/"
          "BORDERLINE 分子的 pKd 与区间仅供参考，最终候选应结合分子对接等"
          "下游实验/计算证据综合裁决。")


if __name__ == "__main__":
    main()
