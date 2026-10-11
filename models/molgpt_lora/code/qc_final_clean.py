# -*- coding: utf-8 -*-
"""
qc_final_clean.py — 按机械规则对生成分子库全量清理，产出最终保留库 + 剔除清单。

规则（硬踢）：
  R1 类金属/重金属原子（Si/B/Sn/As/Se/Al/Ge/Sb/Te/Bi/Hg/Pb/Cd/Tl/Po/Ga/In）
  R2 含P且P的全部邻居无O（游离膦/磷正离子；磷酸酯/氧化膦保留）
  R3 含负电C或负电N原子（异腈[C-]#[N+]一并踢）
  R4 含负电O且不属于白名单（O-邻N+=N-oxide保留；O-邻C(=O)/S/P=盐形式保留并标注；其余醇盐类踢）
  R5 净电荷|q|>1
  R6 MW<150（含环母核碎片如苯/吡啶/苯甲酸；碎片分子对后续口袋对接无意义）
标注（不踢）：
  PAINS/Brenk警报列；isolated_alkene/PAH不踢；
  与阳性对照 Tanimoto>0.85 合规标注（高相似=命中证据而非污染）；
  与训练集 Tc>=0.999 记忆复现标注。

用法
----
    python qc_final_clean.py \
        --input <生成分子库.csv> \
        --control-backup <阳性对照SMILES备份.csv> \
        --train-csv <训练集SMILES表.csv> \
        --out-keep generated_library_final.csv \
        --out-rej rejected_molecules.csv
"""
import argparse
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

METALLOID = {"Si", "B", "Sn", "As", "Se", "Al", "Ge", "Sb", "Te",
             "Bi", "Hg", "Pb", "Cd", "Tl", "Po", "Ga", "In"}  # 覆盖常见类金属与重金属

params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
catalog = FilterCatalog(params)


def fp0(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def main():
    ap = argparse.ArgumentParser(description="生成分子库机械规则全量清理")
    ap.add_argument("--input", required=True, help="生成分子库 CSV（需含 ID、SMILES 列）")
    ap.add_argument("--control-backup", required=True, help="阳性对照 SMILES 备份 CSV（需含 SMILES 列）")
    ap.add_argument("--train-csv", required=True, help="训练集 SMILES 表 CSV")
    ap.add_argument("--out-keep", required=True, help="保留库输出 CSV")
    ap.add_argument("--out-rej", required=True, help="剔除清单输出 CSV")
    args = ap.parse_args()

    pc_smi = pd.read_csv(args.control_backup)["SMILES"].iloc[0]
    pc_mol = Chem.MolFromSmiles(pc_smi)
    pc_fp = fp0(pc_mol)

    train_df = pd.read_csv(args.train_csv)
    train_col = "SMILES" if "SMILES" in train_df.columns else train_df.columns[0]
    train_fps = []
    for s in train_df[train_col]:
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            train_fps.append(fp0(m))

    df = pd.read_csv(args.input)
    keep, rej = [], []
    for _, r in df.iterrows():
        mid, smi = r["ID"], str(r["SMILES"])
        mol = Chem.MolFromSmiles(smi)
        reason, tags = "", []

        if mol is None:
            reason = "R0_parse_fail"
        else:
            mw = Descriptors.MolWt(mol)
            q = Chem.GetFormalCharge(mol)
            has_ring = rdMolDescriptors.CalcNumRings(mol) > 0

            metal = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in METALLOID})
            if metal:
                reason = f"R1_metalloid({','.join(metal)})"
            else:
                # R2: 游离磷（P邻居无O）
                p_no_o = any(
                    a.GetSymbol() == "P"
                    and not any(nb.GetSymbol() == "O" for nb in a.GetNeighbors())
                    for a in mol.GetAtoms()
                )
                neg_cn = [f"{a.GetSymbol()}{a.GetFormalCharge():+d}" for a in mol.GetAtoms()
                          if a.GetFormalCharge() < 0 and a.GetSymbol() in ("C", "N")]
                bad_o = []
                for a in mol.GetAtoms():
                    if a.GetSymbol() == "O" and a.GetFormalCharge() < 0:
                        nbs = [nb.GetSymbol() for nb in a.GetNeighbors()]
                        if "N" in nbs:
                            tags.append("n_oxide")
                        elif a.GetTotalDegree() >= 1 and any(
                            nb.GetSymbol() in ("C", "S", "P")
                            and sum(b.GetBondTypeAsDouble() for b in nb.GetBonds() if b.GetBeginAtomIdx() == a.GetIdx()
                                    or b.GetEndAtomIdx() == a.GetIdx()) >= 1.9
                            for nb in a.GetNeighbors()
                        ):
                            tags.append("salt_form")
                        else:
                            bad_o.append(a.GetIdx())
                if p_no_o:
                    reason = "R2_free_phosphorus"
                elif neg_cn:
                    reason = f"R3_neg_CN({';'.join(neg_cn)})"
                elif bad_o:
                    reason = f"R4_free_alkoxide({len(bad_o)})"
                elif abs(q) > 1:
                    reason = f"R5_net_charge({q:+d})"
                elif mw < 150:
                    reason = f"R6_mw_lt_150({mw:.0f})"

            if not reason and mol is not None:
                entries = catalog.GetMatches(mol)
                alerts = sorted({e.GetDescription() for e in entries})
                if alerts:
                    tags.append("alerts:" + ";".join(alerts))
                sim_pc = DataStructs.TanimotoSimilarity(fp0(mol), pc_fp)
                if sim_pc > 0.85:
                    tags.append(f"posctrl_sim_{sim_pc:.2f}")
                mem = max(DataStructs.TanimotoSimilarity(fp0(mol), t) for t in train_fps) if train_fps else 0
                if mem >= 0.999:
                    tags.append(f"train_mem_{mem:.3f}")

        rec = {"ID": mid, "SMILES": smi, "Source": r.get("Source", ""),
               "MW_orig": r.get("MW", ""), "LogP_orig": r.get("LogP", ""), "TPSA_orig": r.get("TPSA", ""),
               "tags": ";".join(tags)}
        if reason:
            rec["reject_reason"] = reason
            rej.append(rec)
        else:
            keep.append(rec)

    kp, rp = pd.DataFrame(keep), pd.DataFrame(rej)
    kp.to_csv(args.out_keep, index=False, encoding="utf-8-sig")
    rp.to_csv(args.out_rej, index=False, encoding="utf-8-sig")

    print(f"输入: {len(df)} 条")
    print(f"保留: {len(kp)} 条 → {args.out_keep}")
    print(f"剔除: {len(rp)} 条 → {args.out_rej}")
    print("\n剔除原因分布:")
    print(rp["reject_reason"].value_counts().to_string())
    tag_counts = {}
    for t in kp["tags"]:
        for item in [x for x in t.split(";") if x]:
            key = item.split(":")[0] if item.startswith("alerts") else item
            tag_counts[key] = tag_counts.get(key, 0) + 1
    print("\n保留库标注统计:")
    for k, v in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
