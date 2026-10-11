# -*- coding: utf-8 -*-
"""
qc_structural_review.py — 生成分子抽样质检的药物化学结构复审：
基于真实SMILES逐条重算描述符与结构警报（修正抽样质检报告中"结构要点"列
与实际结构可能不符的问题），为人工裁决提供可靠依据。

输入：qc_sample.py 产出的抽样明细 CSV（需含 QC编号、ID、SMILES 列）
输出：逐条描述符 + PAINS/Brenk 警报 + 异常原子 CSV

用法
----
    python qc_structural_review.py \
        --in-csv output/qc/qc_sample_24.csv \
        --out-csv output/qc/structural_review.csv
"""
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# PAINS + Brenk 结构警报目录
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
catalog = FilterCatalog(params)

METALLOID = {"Si", "B", "Sn", "Se", "As", "Al", "Ge", "Sb", "Te"}


def longest_acyclic_carbon_chain(mol):
    """最长无环碳链长度（连续sp3 C，任一起点DFS）"""
    best = 0
    for start in mol.GetAtoms():
        if start.GetSymbol() != "C" or start.IsInRing():
            continue
        stack = [(start.GetIdx(), {start.GetIdx()})]
        while stack:
            idx, seen = stack.pop()
            best = max(best, len(seen))
            a = mol.GetAtomWithIdx(idx)
            for nb in a.GetNeighbors():
                j = nb.GetIdx()
                if j not in seen and nb.GetSymbol() == "C" and not nb.IsInRing():
                    stack.append((j, seen | {j}))
    return best


def main():
    ap = argparse.ArgumentParser(description="抽样分子的药物化学结构复审（描述符重算 + 警报扫描）")
    ap.add_argument("--in-csv", required=True, help="抽样明细 CSV（qc_sample.py 产出）")
    ap.add_argument("--out-csv", required=True, help="输出结构复审 CSV")
    args = ap.parse_args()

    rows = []
    df = pd.read_csv(args.in_csv)
    for _, r in df.iterrows():
        qc, mid, smi = r["QC编号"], r["ID"], r["SMILES"]
        rec = {"QC": qc, "ID": mid, "SMILES": smi}
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            rec.update({"parse_ok": False, "issue": "RDKit解析失败"})
            rows.append(rec)
            continue

        charged = [f"{a.GetSymbol()}{a.GetFormalCharge():+d}" for a in mol.GetAtoms()
                   if a.GetFormalCharge() != 0]
        metalloid = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in METALLOID})
        try:
            entries = catalog.GetMatches(mol)
            alerts = sorted({e.GetDescription() for e in entries})
        except Exception:
            alerts = ["<filter_error>"]

        rings = rdMolDescriptors.CalcNumRings(mol)
        aro_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        rec.update({
            "parse_ok": True,
            "MW": round(Descriptors.MolWt(mol), 1),
            "cLogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 1),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotB": Lipinski.NumRotatableBonds(mol),
            "rings": rings,
            "arom_rings": aro_rings,
            "Fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 2),
            "max_acyclic_chain": longest_acyclic_carbon_chain(mol),
            "formal_charge_atoms": ";".join(charged) if charged else "",
            "net_charge": Chem.GetFormalCharge(mol),
            "metalloid_atoms": ";".join(metalloid) if metalloid else "",
            "pains_brenk": ";".join(alerts) if alerts else "",
            "auto_check_orig": r.get("自动检查", ""),
        })
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    for _, rec in out.iterrows():
        flags = []
        if not rec.get("parse_ok", True):
            flags.append("PARSE_FAIL")
        if rec.get("formal_charge_atoms"):
            flags.append(f"电荷原子[{rec['formal_charge_atoms']}]")
        if rec.get("metalloid_atoms"):
            flags.append(f"类金属[{rec['metalloid_atoms']}]")
        if rec.get("pains_brenk"):
            flags.append(f"警报[{rec['pains_brenk']}]")
        if rec.get("MW", 999) < 150:
            flags.append("MW过小")
        flag_str = " | ".join(flags) if flags else "无硬伤"
        print(f"{rec['QC']}  {rec['ID']}  MW={rec.get('MW','-')}  LogP={rec.get('cLogP','-')}  "
              f"TPSA={rec.get('TPSA','-')}  环={rec.get('rings','-')}/{rec.get('arom_rings','-')}芳  "
              f"最长链={rec.get('max_acyclic_chain','-')}  → {flag_str}")

    print(f"\n已保存: {args.out_csv}")


if __name__ == "__main__":
    main()
