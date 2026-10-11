# -*- coding: utf-8 -*-
"""
qc_sample.py — 生成分子库抽样质检包生成：
1. 计算生成库全部分子与训练集的最大 Tanimoto 相似度（另计与阳性对照的相似度）
2. 三分位分层抽样（高/中/低相似度各 N 条），固定随机种子可复现
3. 自动结构风险检测：超长脂肪链/O-O键/酰氯/硝基数/环数
4. RDKit逐条出PNG + PIL拼总览网格（4列，绕开MolsToGridImage崩溃点）
5. 供人工审查的机器预判数据写入JSON，最终审查文档另由汇总步骤产出

用法
----
    python qc_sample.py \
        --train-csv <训练集SMILES表.csv> \
        --lib-csv <生成分子库.csv> \
        --out-dir output/qc \
        [--control-name positive_control]
"""
import os
import re
import json
import math
import random
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw
RDLogger.DisableLog("rdApp.*")


def longest_chain_C(mol):
    """最长连续sp3 CH2链长度（不 touching杂原子）"""
    max_run = run = 0
    for bond in mol.GetBonds():
        pass
    # 遍历路径法简化：对每个碳原子找连续单键碳链
    adj = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()}
    carbons = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetSymbol() == "C" and a.GetHybridization() == Chem.HybridizationType.SP3]
    def dfs(idx, visited, depth):
        nonlocal max_run
        max_run = max(max_run, depth)
        for n in adj[idx]:
            if n in visited:
                continue
            at = mol.GetAtomWithIdx(n)
            if (at.GetSymbol() == "C" and at.GetHybridization() == Chem.HybridizationType.SP3):
                dfs(n, visited | {n}, depth + 1)
    for c in carbons:
        dfs(c, {c}, 1)
    return max_run


def risk_flags(mol):
    """自动风险检测 → (list of flags)"""
    flags = []
    # 环系统计
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    big_aromatic = sum(1 for r in rings if len(r) >= 5 and
                       all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    total_rings = len(rings)
    if total_rings >= 5:
        flags.append(f"多环堆积({total_rings}环)")
    if total_rings == 0:
        flags.append("无环(链状分子)")
    # O-O键
    for b in mol.GetBonds():
        if b.GetBeginAtom().GetSymbol() == "O" and b.GetEndAtom().GetSymbol() == "O":
            flags.append("含O-O过氧键")
            break
    # 酰氯
    if "C(=O)Cl" in Chem.MolToSmiles(mol) or "ClC=O" in Chem.MolToSmiles(mol):
        flags.append("含酰氯")
    # 硝基数量
    nitro = len(re.findall(r"\[N+\]\(=O\)\[O-\]", Chem.MolToSmiles(mol)))
    if nitro >= 3:
        flags.append(f"硝基x{nitro}")
    # 长脂肪链
    lc = longest_chain_C(mol)
    if lc >= 10:
        flags.append(f"长脂肪链({lc}C)")
    return flags


def main():
    ap = argparse.ArgumentParser(description="生成分子库三分位分层抽样质检")
    ap.add_argument("--train-csv", required=True, help="训练集 SMILES 表 CSV（需含 SMILES、Source 列）")
    ap.add_argument("--lib-csv", required=True, help="生成分子库 CSV（需含 ID、SMILES、MW、LogP、TPSA 列）")
    ap.add_argument("--out-dir", required=True, help="质检输出目录")
    ap.add_argument("--control-name", default="positive_control",
                    help="训练集中阳性对照的 Source 关键词（用于额外计算与其的相似度，找不到则跳过该列）")
    ap.add_argument("--n-per-tier", type=int, default=8, help="每个相似度分层抽取的条数（默认8，共24条）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(42)
    n_per_tier = args.n_per_tier

    print("=" * 60)
    print(f"生成分子库抽样质检：分层 {3 * n_per_tier} 条")
    print("=" * 60)

    # 1. 相似度计算
    df_train = pd.read_csv(args.train_csv)
    train_fps = []
    for s in df_train["SMILES"]:
        m = Chem.MolFromSmiles(s)
        if m:
            train_fps.append((AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024), s))
    ctrl_row = df_train[df_train["Source"].astype(str).str.contains(args.control_name, na=False, regex=False)]
    fp_pc = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(ctrl_row.iloc[0]["SMILES"]), 2, 1024) if len(ctrl_row) else None

    df = pd.read_csv(args.lib_csv)
    print(f"[1] 生成分子库: {len(df)}条，计算相似度...")
    sims_max, sims_pc = [], []
    for smi in df["SMILES"]:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
        sims_max.append(max(DataStructs.TanimotoSimilarity(fp, tfp) for tfp, _ in train_fps))
        sims_pc.append(DataStructs.TanimotoSimilarity(fp, fp_pc) if fp_pc is not None else np.nan)
    df["sim_to_train"] = [round(x, 3) for x in sims_max]
    df["sim_to_posctrl"] = [round(x, 3) if not np.isnan(x) else None for x in sims_pc]

    # 2. 分层抽样
    qs = df["sim_to_train"].quantile([1 / 3, 2 / 3]).tolist()
    tiers = {"高": df[df.sim_to_train > qs[1]], "中": df[(df.sim_to_train > qs[0]) & (df.sim_to_train <= qs[1])],
             "低": df[df.sim_to_train <= qs[0]]}
    picks = []
    for tname, sub in tiers.items():
        idx = list(sub.index)
        random.shuffle(idx)
        for i in idx[:n_per_tier]:
            picks.append((tname, i))
    print(f"[2] 抽样完成: 高(>{qs[1]:.2f})/中({qs[0]:.2f}-{qs[1]:.2f})/低(<{qs[0]:.2f}) 各{n_per_tier}条")

    # 3. 逐条信息+风险检测+画图
    records = []
    png_files = []
    for k, (tier, i) in enumerate(picks):
        row = df.loc[i]
        m = Chem.MolFromSmiles(row["SMILES"])
        flags = risk_flags(m)
        status = "OK" if not flags else "|".join(flags)
        fname = f"qc_{k+1:02d}_{row['ID']}.png"
        fpath = os.path.join(args.out_dir, fname)
        legend = f"{row['ID']}\nMW={row['MW']:.0f} LogP={row['LogP']:.1f}\nsim={row['sim_to_train']}"
        Draw.MolToFile(m, fpath, size=(340, 300), legend=legend)
        png_files.append(fpath)
        records.append({
            "QC编号": f"QC{k+1:02d}",
            "ID": row["ID"],
            "SMILES": row["SMILES"],
            "MW": round(row["MW"], 1), "LogP": round(row["LogP"], 2),
            "TPSA": round(row["TPSA"], 1),
            "sim_to_train": row["sim_to_train"],
            "sim_to_posctrl": row["sim_to_posctrl"],
            "层": tier,
            "自动检查": status or "无异常",
            "环数": m.GetRingInfo().NumRings(),
        })
    qc_df = pd.DataFrame(records)
    qc_csv = os.path.join(args.out_dir, f"qc_sample_{len(records)}.csv")
    qc_df.to_csv(qc_csv, index=False, encoding="utf-8-sig")
    print(f"[3] {len(records)}条明细: {qc_csv}")

    # 4. PIL拼总览（4列，行数按样本量自适应）
    from PIL import Image, ImageDraw, ImageFont
    try:
        windir = os.environ.get("WINDIR", "")
        F_LBL = ImageFont.truetype(os.path.join(windir, "Fonts", "arialbd.ttf"), 22) if windir \
            else ImageFont.load_default()
    except Exception:
        F_LBL = ImageFont.load_default()
    cols = 4
    rows_n = max(1, math.ceil(len(png_files) / cols))
    cw, ch, pad, hdr = 350, 310, 12, 50
    W = cols * cw + (cols + 1) * pad
    H = rows_n * ch + (rows_n + 1) * pad + hdr
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)
    dr.text((pad, 12), f"Generated library QC sample: {len(png_files)} molecules "
                       "(tiered by similarity to target ligand set)",
            font=F_LBL, fill="black")
    for k, fp in enumerate(png_files):
        img = Image.open(fp).convert("RGB").resize((cw - 10, ch - 10))
        c, r_ = k % cols, k // cols
        x = pad + c * (cw + pad) + 5
        y = hdr + pad + r_ * (ch + pad) + 5
        canvas.paste(img, (x, y))
    overview = os.path.join(args.out_dir, "qc_overview_grid.png")
    canvas.save(overview)
    print(f"[4] 总览图: {overview}")

    # 统计摘要
    print(f"\n[5] 自动检查异常统计:")
    flag_cnt = {}
    for rec in records:
        if rec["自动检查"] != "无异常":
            for fl in rec["自动检查"].split("|"):
                key = fl.split("(")[0]
                flag_cnt[key] = flag_cnt.get(key, 0) + 1
    for k_, v_ in sorted(flag_cnt.items(), key=lambda x: -x[1]):
        print(f"   {k_}: {v_}/{len(records)}")
    clean = sum(1 for rec in records if rec["自动检查"] == "无异常")
    print(f"   完全无异常: {clean}/{len(records)}")

    with open(os.path.join(args.out_dir, "_records.json"), "w", encoding="utf-8") as f:
        json.dump({"tiers": {"high": f">{qs[1]:.3f}", "mid": f"{qs[0]:.3f}-{qs[1]:.3f}", "low": f"<{qs[0]:.3f}"},
                   "records": records}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
