# -*- coding: utf-8 -*-
"""
微调数据集构建：剔除阳性对照 + 可选补入文献种子配体
=====================================================================
逻辑：
- 输入：靶点公开已知配体表 CSV（列：SMILES, Source, max_sim 可选），几十条量级；
- 阳性对照分子剔除（作为外部验证锚点，绝不进入训练集）；
- 可选补入文献报道的种子配体（--extra-seeds 传入 CSV：SMILES,Source 两列）；
- 产出：finetune_smiles.csv（不含阳性对照）。

阳性对照三重隔离硬断言（结构级兜底，独立于 Source 标签）：
  仅凭 Source 字符串剔除可能被标签笔误绕过，这里用分子结构本身做
  三路独立校验（Source 标签 / canonical SMILES / 去立体 SMILES），
  任一命中即中止、拒绝写出训练集。

用法：
  python build_finetune_dataset.py \
      --input known_ligands.csv \
      --control-name positive_control \
      [--extra-seeds extra_seeds.csv]
"""
import argparse
import os
import sys
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")


def main():
    ap = argparse.ArgumentParser(description="微调数据集构建（剔除阳性对照）")
    ap.add_argument("--input", required=True, help="已知配体表 CSV（SMILES,Source[,max_sim]）")
    ap.add_argument("--output", default="finetune_smiles.csv", help="输出训练集 CSV")
    ap.add_argument("--control-backup", default="positive_control_backup.csv",
                    help="阳性对照单独备份 CSV（作为外部验证锚点）")
    ap.add_argument("--control-name", default="positive_control",
                    help="Source 列中标识阳性对照的关键字（子串匹配）")
    ap.add_argument("--control-id", default="",
                    help="阳性对照的公开数据库 CID/编号（可选，用于标签级断言；留空跳过）")
    ap.add_argument("--extra-seeds", default="",
                    help="可选：额外种子配体 CSV（SMILES,Source 两列），逐条校验后并入训练集")
    args = ap.parse_args()

    print("=" * 60)
    print("微调数据集构建：剔除阳性对照 + 可选补入文献种子配体")
    print("=" * 60)

    df = pd.read_csv(args.input)
    print(f"[1] 原数据集: {len(df)}条")

    # 校验额外种子分子（如某文献报道的已知抑制剂）
    seed_rows = []
    if args.extra_seeds:
        for _, row in pd.read_csv(args.extra_seeds).iterrows():
            mol = Chem.MolFromSmiles(str(row["SMILES"]))
            if mol is None:
                print(f"  ❌ 种子 SMILES 解析失败: {row['Source']}")
                sys.exit(1)
            print(f"  种子校验: {row['Source']} MW={Descriptors.MolWt(mol):.1f}, "
                  f"重原子={mol.GetNumHeavyAtoms()}")
            seed_rows.append({"SMILES": str(row["SMILES"]), "Source": str(row["Source"]),
                              "max_sim": 1.0})

    # 分离阳性对照与保留配体
    ctrl_mask = df["Source"].astype(str).str.contains(args.control_name, na=False)
    ctrl_rows = df[ctrl_mask]
    others = df[~ctrl_mask].copy()

    print(f"  阳性对照: {len(ctrl_rows)}条 (剔除→外部验证锚点)")
    print(f"  已知配体: {len(others)}条 (保留)")
    if ctrl_rows.empty:
        print("  ⚠️ 未在 Source 中匹配到阳性对照，请确认 --control-name 设置")

    # 备份阳性对照
    ctrl_rows.to_csv(args.control_backup, index=False, encoding="utf-8-sig")
    print(f"  阳性对照备份: {args.control_backup}")

    # 构建训练集: 已知配体 + 额外种子
    parts = [others]
    if seed_rows:
        parts.append(pd.DataFrame(seed_rows))
    df_v2 = pd.concat(parts, ignore_index=True)
    if "max_sim" in df_v2.columns:
        df_v2["max_sim"] = df_v2["max_sim"].fillna(1.0)

    # 验证每条SMILES
    valid = 0
    for _, row in df_v2.iterrows():
        m = Chem.MolFromSmiles(str(row["SMILES"]))
        if m:
            valid += 1
        else:
            print(f"  ⚠️ 解析失败: {str(row['Source'])[:30]}")
    print(f"[3] 训练集验证: {valid}/{len(df_v2)}条有效")

    # ===== 阳性对照三重隔离硬断言（结构级兜底，独立于 Source 标签）=====
    # 目的：阳性对照仅作外部验证锚点，绝不可进入训练集。
    # 结构级独立校验，任一失败即中止、拒绝写出。
    def _can(smi, strip_stereo=False):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        if strip_stereo:
            Chem.RemoveStereochemistry(m)  # 去立体后比对，防止手性写法差异绕过
        return Chem.MolToSmiles(m, canonical=True)

    if not ctrl_rows.empty:
        ctrl_smi_guard = str(ctrl_rows.iloc[0]["SMILES"])
        ctrl_can = _can(ctrl_smi_guard)
        ctrl_nostereo = _can(ctrl_smi_guard, strip_stereo=True)
        v2_can = {_can(s) for s in df_v2["SMILES"].astype(str)}
        v2_nostereo = {_can(s, strip_stereo=True) for s in df_v2["SMILES"].astype(str)}
        v2_source = " ".join(df_v2["Source"].astype(str))

        guard = {
            "cid_absent": (args.control_id not in v2_source) if args.control_id else True,
            "name_absent": args.control_name not in v2_source,
            "canonical_smiles_absent": ctrl_can not in v2_can,
            "nostereo_smiles_absent": ctrl_nostereo not in v2_nostereo,
        }
        failed = [k for k, ok in guard.items() if not ok]
        if failed:
            raise RuntimeError(f"阳性对照隔离断言失败，拒绝写出训练集: {failed}")
        print(f"[3b] 阳性对照三重隔离硬断言 PASS: {list(guard.keys())}")

    df_v2.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[4] 训练集输出: {args.output}")

    # 写后读回再验一次（防写出环节异常）
    df_chk = pd.read_csv(args.output)
    if not ctrl_rows.empty:
        chk_nostereo = {_can(s, strip_stereo=True) for s in df_chk["SMILES"].astype(str)}
        chk_can = {_can(s) for s in df_chk["SMILES"].astype(str)}
        chk_source = " ".join(df_chk["Source"].astype(str))
        if (ctrl_can in chk_can or ctrl_nostereo in chk_nostereo
                or (args.control_id and args.control_id in chk_source)):
            raise RuntimeError("写后校验失败: 阳性对照出现在最终训练集文件中!")
        print(f"[4b] 写后读回校验 PASS: {len(df_chk)}条, 阳性对照确认不在其中")
    print(f"  已知配体: {len(others)}条")
    print(f"  额外种子: {len(seed_rows)}条")
    print(f"  外部验证锚点: 阳性对照 ({len(ctrl_rows)}条, 不在训练集中)")

    # 与阳性对照的相似度（检查种子配体与对照是否同族，供人工参考）
    if (not ctrl_rows.empty) and seed_rows:
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        ctrl_mol = Chem.MolFromSmiles(str(ctrl_rows.iloc[0]["SMILES"]))
        ctrl_fp = AllChem.GetMorganFingerprintAsBitVect(ctrl_mol, 2, 1024)
        print(f"\n[5] 种子-阳性对照相似度验证:")
        for srow in seed_rows:
            smol = Chem.MolFromSmiles(srow["SMILES"])
            sfp = AllChem.GetMorganFingerprintAsBitVect(smol, 2, 1024)
            sim = DataStructs.TanimotoSimilarity(ctrl_fp, sfp)
            print(f"  {srow['Source'][:40]}-对照 Tanimoto: {sim:.3f} "
                  f"({'同族' if sim > 0.3 else '异族'})")


if __name__ == "__main__":
    main()
