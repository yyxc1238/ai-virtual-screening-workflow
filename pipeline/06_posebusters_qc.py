#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""06 构象物理规则质检脚本（参考工作流文档 05 篇 2.3 节"构象物理规则门"）

原理（文档 05 篇）：对接打分函数是经验拟合的，不保证构象本身物理成立；采样器会产出
穿模、自撞、键长/键角失真的"假构象"且可能拿到高分。用 PoseBusters 的物理规则集
（约 19 项检查）自动判定构象是否成立，把"人工全检"变成"自动门 + 边界案例抽检"。

功能：
1. 对批量 pose（SDF，每行一个构象）逐个运行 PoseBusters 检查；
2. 汇总 PASS/FAIL 与失败类型分布表（CSV），便于定位系统性制备问题；
3. 异常识别（文档 05 篇）：物理门淘汰率异常高（如 >25%）时提示检查受体/配体制备
   （常是加氢、质子化或格式问题）。

依赖：posebusters、rdkit、pandas
  安装：pip install posebusters rdkit pandas
用法示例：
    python 06_posebusters_qc.py --poses docked_poses.sdf --out-dir qc_out
    python 06_posebusters_qc.py --poses poses.sdf --receptor receptor.pdb --out-dir qc_out
"""
import argparse
import json
import os
import sys
import time

try:
    from rdkit import Chem
    import pandas as pd
except ImportError as e:
    sys.exit("[错误] 缺少依赖：%s\n安装：pip install rdkit pandas" % e)


def load_reference(path):
    """读取参考构象（SDF/MOL 取第一个分子；PDB 直接解析）。"""
    if path.lower().endswith(".pdb"):
        return Chem.MolFromPDBFile(path, removeHs=False)
    mols = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    return mols[0] if mols else None


def main():
    ap = argparse.ArgumentParser(
        description="PoseBusters 构象物理规则质检与失败类型分布统计，参考工作流文档 05 篇 2.3 节",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", required=True, help="pose 文件（SDF，多构象；来自 05 脚本对接输出）")
    ap.add_argument("--receptor", help="受体文件（PDB/PDBQT）；提供后启用含受体的 dock/redock 检查")
    ap.add_argument("--reference", help="参考配体（共晶构象 SDF/PDB）；与受体同时提供时用 redock 配置")
    ap.add_argument("--config", choices=["auto", "mol", "dock", "redock"], default="auto",
                    help="PoseBusters 规则配置（默认 auto：有参考+受体用 redock，仅受体用 dock，否则 mol）")
    ap.add_argument("--fail-rate-warn", type=float, default=0.25,
                    help="淘汰率告警阈值（默认 0.25，即 25%%；文档 05 篇异常识别口径）")
    ap.add_argument("--out-dir", default="posebusters_qc", help="输出目录（默认 posebusters_qc）")
    args = ap.parse_args()

    try:
        from posebusters import PoseBusters
    except ImportError:
        sys.exit("[错误] 缺少 posebusters。安装：pip install posebusters")

    poses = [m for m in Chem.SDMolSupplier(args.poses, removeHs=False) if m is not None]
    if not poses:
        sys.exit("[错误] 未从 %s 解析到 pose 分子" % args.poses)

    receptor = args.receptor if (args.receptor and os.path.isfile(args.receptor)) else None
    reference = load_reference(args.reference) if (args.reference and os.path.isfile(args.reference)) else None
    if args.config == "auto":
        config = "redock" if (reference is not None and receptor) else ("dock" if receptor else "mol")
    else:
        config = args.config
    print("[info] pose %d 个，规则配置 = %s" % (len(poses), config))

    pb = PoseBusters(config=config)
    rows = []
    for i, mol in enumerate(poses):
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "pose_%d" % i
        try:
            if config == "mol":
                df = pb.bust(mol)
            else:
                df = pb.bust(mol, reference, receptor)
        except Exception as exc:
            rows.append({"pose_index": i, "name": name, "config": config,
                         "pass": False, "failed_checks": "runtime_error: %s" % exc})
            continue
        bool_cols = [c for c in df.columns if df[c].dtype == bool]
        vals = df.iloc[0]
        failed = [c for c in bool_cols if not bool(vals[c])]
        row = {"pose_index": i, "name": name, "config": config,
               "pass": (not failed), "failed_checks": ";".join(failed)}
        for extra in ("rmsd", "centroid_distance", "minimum_distance", "volume_overlap"):
            if extra in df.columns:
                row[extra] = vals[extra]
        rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    df_rows = pd.DataFrame(rows)
    per_pose_csv = os.path.join(args.out_dir, "posebusters_results.csv")
    df_rows.to_csv(per_pose_csv, index=False, encoding="utf-8-sig")

    # ---- 失败类型分布表（哪条规则最常触发 -> 定位制备问题）----
    fail_counts = Counter_like = {}
    for r in rows:
        for c in (r.get("failed_checks") or "").split(";"):
            c = c.strip()
            if c and not c.startswith("runtime_error"):
                fail_counts[c] = fail_counts.get(c, 0) + 1
    dist_rows = sorted(fail_counts.items(), key=lambda kv: -kv[1])
    with open(os.path.join(args.out_dir, "failure_distribution.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("check,n_failed,fail_rate\n")
        for check, cnt in dist_rows:
            f.write("%s,%d,%.4f\n" % (check, cnt, cnt / max(len(rows), 1)))

    n_pass = sum(1 for r in rows if r["pass"])
    n_fail = len(rows) - n_pass
    fail_rate = n_fail / max(len(rows), 1)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "poses_file": os.path.basename(args.poses), "config": config,
        "n_total": len(rows), "n_pass": n_pass, "n_fail": n_fail,
        "fail_rate": round(fail_rate, 4), "fail_rate_warn_threshold": args.fail_rate_warn,
        "failure_counts": dict(dist_rows),
        "note": "对接打分只在物理成立的构象里才有排序意义（文档 05 篇 2.3 节）；"
                "PASS 的 pose 才进入后续精排/融合",
    }
    with open(os.path.join(args.out_dir, "qc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[台账] 总数 %d | PASS %d | FAIL %d（淘汰率 %.1f%%）" % (len(rows), n_pass, n_fail, fail_rate * 100))
    for check, cnt in dist_rows[:8]:
        print("  失败规则 %-32s %d" % (check, cnt))
    if fail_rate > args.fail_rate_warn:
        print("[警告] 物理门淘汰率 %.1f%% 超过 %.0f%%：请检查受体/配体制备（加氢、质子化、格式），"
              "文档 05 篇异常识别" % (fail_rate * 100, args.fail_rate_warn * 100))
    print("[输出] 逐 pose 结果与失败类型分布已写入 %s" % args.out_dir)


if __name__ == "__main__":
    main()
