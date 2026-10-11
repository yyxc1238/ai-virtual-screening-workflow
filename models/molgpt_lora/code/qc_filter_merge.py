# -*- coding: utf-8 -*-
"""
qc_filter_merge.py — 生成分子四道关卡过滤 → 与既有参考库去重后并入生成分子库

关卡1 Lipinski:  MW<500, LogP<5, HBD<=5, HBA<=10
关卡2 Veber:     TPSA<=140, 可旋转键<=10
关卡3 PAINS:     rdkit FilterCatalog 告警排除
关卡4 SA评分:    Ertl SAScore < 4.5（易合成）

用法
----
    python qc_filter_merge.py \
        --gen-smi output/generated_finetuned.smi \
        --source-lib <既有参考库.csv> \
        --out-csv generated_library_merged.csv
"""
import argparse
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, Crippen
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog("rdApp.*")

# SA Score（rdkit内置Contrib）
try:
    from rdkit.Contrib.SA_Score import sascorer
    HAS_SA = True
except Exception as e:
    print(f"[WARN] SA_Score不可用: {e}")
    HAS_SA = False

# PAINS目录
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog = FilterCatalog(params)


def props(m):
    return {
        "MW": Descriptors.MolWt(m),
        "LogP": Crippen.MolLogP(m),
        "HBD": Lipinski.NumHDonors(m),
        "HBA": Lipinski.NumHAcceptors(m),
        "TPSA": Descriptors.TPSA(m),
        "RotB": Lipinski.NumRotatableBonds(m),
    }


def passes(m):
    p = props(m)
    r1 = p["MW"] < 500 and p["LogP"] < 5 and p["HBD"] <= 5 and p["HBA"] <= 10
    r2 = p["TPSA"] <= 140 and p["RotB"] <= 10
    r3 = not catalog.HasMatch(m)
    r4 = (sascorer.calculateScore(m) < 4.5) if HAS_SA else True
    return r1 and r2 and r3 and r4, {"lipinski": int(r1), "veber": int(r2),
                                     "no_pains": int(r3), "sa_ok": int(r4), **{k: round(v, 2) for k, v in p.items()}}


def main():
    ap = argparse.ArgumentParser(description="生成分子四道关卡过滤，与既有参考库去重后并入生成分子库")
    ap.add_argument("--gen-smi", required=True, help="微调模型生成的 SMILES 文件（每行一条）")
    ap.add_argument("--source-lib", required=True, help="既有参考库 CSV（需含 SMILES 列），用于去重")
    ap.add_argument("--out-csv", required=True, help="输出合并后的生成分子库 CSV")
    args = ap.parse_args()

    print("=" * 60)
    print("四道关卡过滤 + 并入生成分子库")
    print("=" * 60)

    # 1. 读生成分子
    gen = [l.strip() for l in open(args.gen_smi, encoding="utf-8") if l.strip()]
    print(f"[1] 微调生成分子: {len(gen)}条")

    # 2. 四道关卡
    rows, funnel = [], {"input": len(gen), "lipinski": 0, "veber": 0, "no_pains": 0, "sa_ok": 0, "final": 0}
    for smi in gen:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        ok, flags = passes(m)
        for k in ("lipinski", "veber", "no_pains", "sa_ok"):
            funnel[k] += flags[k]
        if ok:
            funnel["final"] += 1
            rows.append({"SMILES": Chem.MolToSmiles(m), "MW": flags["MW"], "LogP": flags["LogP"],
                         "TPSA": flags["TPSA"], "SA": round(sascorer.calculateScore(m), 2) if HAS_SA else None})
    print(f"[2] 漏斗: 输入{funnel['input']} -> Lipinski通过{funnel['lipinski']} "
          f"-> +Veber {funnel['veber']} -> 无PAINS {funnel['no_pains']} -> SA合格 {funnel['final']}")

    # 3. 与既有库去重
    old = pd.read_csv(args.source_lib)
    old_set = set()
    for s in old.get("SMILES", pd.Series(dtype=str)):
        mm = Chem.MolFromSmiles(str(s))
        if mm:
            old_set.add(Chem.MolToSmiles(mm))
    print(f"[3] 既有库: {len(old)}条 (canonical unique {len(old_set)})")
    before = len(rows)
    rows = [r for r in rows if r["SMILES"] not in old_set]
    print(f"    与既有库去重: {before} -> {len(rows)} (新增真正 novel {len(rows)}条)")

    # 4. 输出生成分子库
    out = pd.DataFrame(rows)
    out.insert(0, "ID", [f"GEN_{i+1:04d}" for i in range(len(out))])
    out.insert(2, "Source", "MolGPT_LoRA_finetuned")
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[4] 生成分子库输出: {args.out_csv}")
    print(f"    库规模: {len(out)}条生成新分子")

    # 5. 汇总
    if len(out):
        print(f"    MW: {out['MW'].min():.0f}-{out['MW'].max():.0f} (median {out['MW'].median():.0f})")
        print(f"    LogP: {out['LogP'].min():.1f}-{out['LogP'].max():.1f}")
        if HAS_SA and out["SA"].notna().any():
            print(f"    SA: {out['SA'].min():.1f}-{out['SA'].max():.1f} (median {out['SA'].median():.1f})")
    print("\n过滤合并完成！")


if __name__ == "__main__":
    main()
