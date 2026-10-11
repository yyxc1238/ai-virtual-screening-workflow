#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""07 多证据秩融合与可开发性提醒脚本（参考工作流文档 08 篇《多证据融合与可开发性门》）

功能：
1. 多证据秩融合（文档 2.1 节）：不同来源的分数（如 vina 对接分、自研模型 pKd、
   AI 初筛分）尺度与方向各异，不能直接相加；先把每个指标在候选集内转为
   归一化排名分（0-1），再加权求和得到融合分。权重经 --weights 指定（默认均等），
   应在看到结果之前确定并冻结（文档 2.1 节"权重冻结"）；
2. 模型分只投票、不设硬门槛（文档 2.2 节）：本脚本不做任何硬淘汰；
3. 简易可开发性提醒列（文档 2.3 节第④维的粗略代理）：用 RDKit 计算 MW 与
   Crippen LogP，对超过阈值者打标。完整四维评分（可得性/药理先例/通路合理性/
   安全类药性）需人工或 ADMET 工具补齐，不在本脚本范围内。

依赖：pandas、rdkit
  安装：pip install pandas rdkit
用法示例：
    python 07_fusion_rank.py --input scores.csv --scores vina,affinity_model,ai_screen \
        --lower-is-better vina --weights 0.5,0.3,0.2 --smiles-col smiles --out fusion_rank.csv
"""
import argparse
import os
import sys

try:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors
except ImportError as e:
    sys.exit("[错误] 缺少依赖：%s\n安装：pip install pandas rdkit" % e)


def split_list(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def main():
    ap = argparse.ArgumentParser(
        description="多证据秩归一化加权融合排名 + 简易可开发性提醒，参考工作流文档 08 篇",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="候选分数表 CSV（每行一个候选，含多个分数列）")
    ap.add_argument("--id-col", default="name", help="候选标识列（默认 name；不存在时用第一列）")
    ap.add_argument("--scores", required=True, help="参与融合的分数列名，逗号分隔（如 vina,affinity_model）")
    ap.add_argument("--lower-is-better", default="",
                    help="其中『越小越好』的列名，逗号分隔（如对接分；未列出的按『越大越好』处理）")
    ap.add_argument("--weights", default="",
                    help="各分数权重，逗号分隔（默认均等；应在看到结果前冻结，文档 08 篇 2.1 节）")
    ap.add_argument("--smiles-col", help="SMILES 列名（提供时计算 MW/LogP 可开发性提醒）")
    ap.add_argument("--mw-max", type=float, default=500.0,
                    help="分子量提醒阈值（默认 500，Lipinski 口径，可按项目调整）")
    ap.add_argument("--logp-max", type=float, default=5.0,
                    help="LogP 提醒阈值（默认 5，Lipinski 口径，可按项目调整）")
    ap.add_argument("--out", default="fusion_rank.csv", help="输出 CSV 路径")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    score_cols = split_list(args.scores)
    lower_cols = set(split_list(args.lower_is_better))
    missing = [c for c in score_cols + [args.id_col] if c not in df.columns]
    if missing:
        sys.exit("[错误] 输入表缺少列：%s（现有列：%s）" % (missing, list(df.columns)))
    if len(df) == 0:
        sys.exit("[错误] 输入表为空")

    # ---- 权重解析与归一化（文档 08 篇：权重代表各证据的话语权，融合前冻结）----
    weights = [float(x) for x in split_list(args.weights)] if args.weights else [1.0] * len(score_cols)
    if len(weights) != len(score_cols):
        sys.exit("[错误] --weights 数量（%d）与 --scores 数量（%d）不一致" % (len(weights), len(score_cols)))
    w_sum = sum(weights)
    if w_sum <= 0:
        sys.exit("[错误] 权重之和必须为正")
    weights = [w / w_sum for w in weights]

    # ---- 秩归一化：每列转为 0-1 排名分，方向统一为"越大越好"（文档 2.1 节）----
    fusion = pd.Series(0.0, index=df.index)
    for col, w in zip(score_cols, weights):
        ascending = col not in lower_cols   # ascending=True 时值越大 pct 越大（即越好）
        norm = df[col].rank(pct=True, ascending=ascending)
        norm = norm.fillna(0.0)             # 缺失分按最差处理，避免整行丢弃
        df[col + "_norm"] = norm.round(6)
        fusion = fusion + w * norm
    df["fusion_score"] = fusion.round(6)
    df["fusion_rank"] = df["fusion_score"].rank(ascending=False, method="min").astype(int)

    # ---- 简易可开发性提醒（文档 08 篇 2.3 节第④维的粗略代理，仅提醒、不淘汰）----
    if args.smiles_col:
        if args.smiles_col not in df.columns:
            sys.exit("[错误] 输入表缺少 SMILES 列：%s" % args.smiles_col)
        mw_list, logp_list, warn_list = [], [], []
        for smi in df[args.smiles_col]:
            m = Chem.MolFromSmiles(str(smi))
            if m is None:
                mw_list.append(float("nan"))
                logp_list.append(float("nan"))
                warn_list.append("invalid_smiles")
                continue
            mw = Descriptors.MolWt(m)
            logp = Crippen.MolLogP(m)
            warns = []
            if mw > args.mw_max:
                warns.append("MW>%.0f" % args.mw_max)
            if logp > args.logp_max:
                warns.append("LogP>%.0f" % args.logp_max)
            mw_list.append(mw)
            logp_list.append(logp)
            warn_list.append(";".join(warns) if warns else "")
        df["MW"] = [round(v, 2) for v in mw_list]
        df["LogP"] = [round(v, 2) for v in logp_list]
        df["dev_warning"] = warn_list

    df = df.sort_values("fusion_rank")
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("[完成] 融合排名已写入 %s（候选 %d 个）" % (args.out, len(df)))
    print("[权重] %s" % ", ".join("%s=%.3f" % (c, w) for c, w in zip(score_cols, weights)))
    if args.smiles_col:
        n_warn = int((df["dev_warning"].fillna("") != "").sum())
        print("[提醒] 可开发性预警 %d 个（MW>%.0f 或 LogP>%.0f；完整四维评分见文档 08 篇 2.3 节）"
              % (n_warn, args.mw_max, args.logp_max))
    print("[提示] 融合只影响排序，不做硬淘汰；模型分不可靠（分布外）时以物理证据为主（文档 08 篇 2.2 节）")
    print("[预览] 前 5 名：")
    for _, r in df.head(5).iterrows():
        print("  #%d %s fusion=%.4f" % (r["fusion_rank"], r[args.id_col], r["fusion_score"]))


if __name__ == "__main__":
    main()
