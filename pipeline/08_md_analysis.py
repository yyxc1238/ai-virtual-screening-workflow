#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""08 MD 轨迹机械裁决分析脚本（参考工作流文档 09 篇 2.5 节 B1-B4 判据）

功能：读取 GROMACS 产物（.xtc/.trr/.nc 等轨迹 + 拓扑），用 MDAnalysis 对每条
轨迹（对应一个 pose 的短 MD）计算四项预冻结判据：

  B1 配体稳定度：配体（重原子）相对初始 pose 的 RMSD（蛋白主链叠合后）
                 -> PASS 若最大值 <= --b1-rmsd（默认 5.0 A，可调）
  B2 接触维持率：初始口袋接触残基在轨迹中被维持的比例（全程平均）
                 -> PASS 若均值 > --b2-min（默认 0.5）
  B3 配体位置：  配体质心相对口袋参考点（默认初始质心）的漂移
                 -> PASS 若"漂移超过 --b3-drift（默认 5.0 A）的帧占比" < --b3-max-frac（默认 0.2）
  B4 物理异常：  位点残基骨架 RMSD 是否出现持续性大幅异常偏离
                 （参考阈值约 4 A，文档 09 篇；区分孤立瞬时尖峰与长期超限）
                 -> PASS 若最大值 <= --b4-rmsd，或超限帧占比 <= --b4-spike-frac

分子级裁决（文档 09 篇 2.5 节）：每个分子取 top 代表 pose 各跑短 MD，
任一 pose 四条全 PASS 则该分子存活（SURVIVE）；全部 pose 失败才淘汰（ELIMINATE）。
输出机器可读 verdict JSON，逐 pose 留档可溯源。未部署 MD 的分子必须显式标注，
不能暗示做过（文档 09 篇验收标准）。

依赖：MDAnalysis、numpy（安装：pip install MDAnalysis numpy）
用法示例：
    python 08_md_analysis.py --molecule cand01 --topol md.tpr \
        --traj pose1.xtc --traj pose2.xtc --pose-label pose1 --pose-label pose2 \
        --ligand-sel "resname LIG and not name H*" \
        --site-sel "protein and around 6 of resname LIG" --out verdict_cand01.json
"""
import argparse
import json
import os
import sys
import time

try:
    import numpy as np
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array
    from MDAnalysis.analysis.rms import rotation_matrix
except ImportError as e:
    sys.exit("[错误] 缺少依赖：%s\n安装：pip install MDAnalysis numpy" % e)


def freeze(universe, selection):
    """按第 0 帧的选择结果冻结静态原子组（避免 around 等动态选择逐帧变化）。"""
    universe.trajectory[0]
    grp = universe.select_atoms(selection)
    if grp.n_atoms == 0:
        sys.exit("[错误] 选择 '%s' 命中 0 个原子，请检查选择字符串" % selection)
    return universe.atoms[grp.indices]


def analyze_pose(topol, traj, args):
    """对单条轨迹（一个 pose 的短 MD）计算 B1-B4 指标。"""
    u = mda.Universe(topol, traj)
    lig = freeze(u, args.ligand_sel)      # 默认仅重原子（RMSD/质心均基于重原子）
    fit = freeze(u, args.fit_sel)         # 全局叠合参考组（默认全蛋白主链）
    site = freeze(u, args.site_sel)       # 口袋/位点残基（B2 接触 + B4 骨架）
    bb_sel = args.bb_sel or ("(%s) and backbone" % args.site_sel)
    bb = freeze(u, bb_sel)

    ref_fit = fit.positions.astype(np.float64)
    ref_lig = lig.positions.astype(np.float64)
    ref_bb = bb.positions.astype(np.float64)
    site_resids = site.resids
    ref_com = lig.center_of_mass()
    pocket_ref = np.array(args.pocket_center, dtype=np.float64) if args.pocket_center else ref_com

    # 初始接触残基集合（第 0 帧，距离 <= cutoff）
    d0 = distance_array(ref_lig, site.positions.astype(np.float64))
    init_contacts = set(site_resids[d0.min(axis=1) <= args.b2_contact_cutoff].tolist())

    b1_list, b2_list, b3_list, b4_list = [], [], [], []
    n_frames = 0
    for _ts in u.trajectory[::args.stride]:
        n_frames += 1
        cur_fit = fit.positions.astype(np.float64)
        rot, _rms = rotation_matrix(cur_fit, ref_fit)
        c_cur = cur_fit.mean(axis=0)
        c_ref = ref_fit.mean(axis=0)

        # B1：叠合后配体重原子 RMSD（相对初始 pose）
        lig_al = (lig.positions.astype(np.float64) - c_cur) @ rot.T + c_ref
        b1 = float(np.sqrt(np.mean(np.sum((lig_al - ref_lig) ** 2, axis=1))))
        b1_list.append(b1)

        # B4：全局叠合后位点残基骨架 RMSD（相对初始帧）
        bb_al = (bb.positions.astype(np.float64) - c_cur) @ rot.T + c_ref
        b4 = float(np.sqrt(np.mean(np.sum((bb_al - ref_bb) ** 2, axis=1))))
        b4_list.append(b4)

        # B3：配体质心相对口袋参考点的漂移
        com = lig.center_of_mass()
        b3 = float(np.linalg.norm(com - pocket_ref))
        b3_list.append(b3)

        # B2：初始接触残基的维持比例
        if init_contacts:
            d = distance_array(lig.positions.astype(np.float64),
                               site.positions.astype(np.float64))
            contacted = set(site_resids[d.min(axis=1) <= args.b2_contact_cutoff].tolist())
            b2_list.append(len(contacted & init_contacts) / len(init_contacts))
        else:
            b2_list.append(float("nan"))

    b1_max = max(b1_list)
    b2_mean = float(np.nanmean(b2_list)) if any(not np.isnan(v) for v in b2_list) else float("nan")
    b3_frac = float(np.mean(np.asarray(b3_list) > args.b3_drift))
    b4_max = max(b4_list)
    b4_over_frac = float(np.mean(np.asarray(b4_list) > args.b4_rmsd))

    b1_pass = b1_max <= args.b1_rmsd
    b2_pass = (not np.isnan(b2_mean)) and b2_mean > args.b2_min
    b3_pass = b3_frac < args.b3_max_frac
    # 文档 09 篇 B4：区分孤立瞬时尖峰与长期超限
    b4_pass = (b4_max <= args.b4_rmsd) or (b4_over_frac <= args.b4_spike_frac)

    warnings = []
    if not init_contacts:
        warnings.append("第 0 帧无初始接触：请检查 --ligand-sel / --site-sel 是否正确，B2 记为 FAIL")
    return {
        "topology": os.path.basename(topol), "trajectory": os.path.basename(traj),
        "n_frames_analyzed": n_frames, "stride": args.stride,
        "n_initial_contact_residues": len(init_contacts),
        "B1_ligand_rmsd_max_A": round(b1_max, 3),
        "B2_contact_maintenance": None if np.isnan(b2_mean) else round(b2_mean, 4),
        "B3_drift_over_threshold_frac": round(b3_frac, 4),
        "B3_drift_max_A": round(max(b3_list), 3),
        "B4_site_backbone_rmsd_max_A": round(b4_max, 3),
        "B4_over_threshold_frac": round(b4_over_frac, 4),
        "B1_pass": b1_pass, "B2_pass": b2_pass, "B3_pass": b3_pass, "B4_pass": b4_pass,
        "overall": bool(b1_pass and b2_pass and b3_pass and b4_pass),
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser(
        description="MD 轨迹 B1-B4 机械裁决分析（MDAnalysis），参考工作流文档 09 篇",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--molecule", required=True, help="分子标识（写入 verdict JSON）")
    ap.add_argument("--topol", required=True, help="拓扑文件（GROMACS .tpr / .gro / .pdb 等）")
    ap.add_argument("--traj", action="append", required=True,
                    help="轨迹文件（.xtc/.trr 等；一个 pose 一条轨迹，可多次指定）")
    ap.add_argument("--pose-label", action="append", default=[],
                    help="与 --traj 一一对应的 pose 标签（缺省自动命名 pose1..N）")
    ap.add_argument("--ligand-sel", default="resname LIG and not name H*",
                    help="配体选择（默认重原子 'resname LIG and not name H*'；GROMACS 常用 LIG 命名）")
    ap.add_argument("--fit-sel", default="protein and backbone",
                    help="全局叠合参考选择（默认全蛋白主链）")
    ap.add_argument("--site-sel", default="protein",
                    help="口袋/位点残基选择（B2 接触与 B4 骨架用；建议如 'protein and around 6 of resname LIG'）")
    ap.add_argument("--bb-sel", default=None,
                    help="B4 骨架选择（默认由 --site-sel 派生：'(...) and backbone'）")
    ap.add_argument("--pocket-center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="B3 漂移参考点（默认初始配体质心）")
    # ---- B1-B4 阈值：看结果前冻结，逐 pose 可溯源（文档 09 篇 2.5 节）----
    ap.add_argument("--b1-rmsd", type=float, default=5.0,
                    help="B1 配体 RMSD 上限（A，默认 5.0；文档 09 篇未定统一数值，按项目冻结）")
    ap.add_argument("--b2-min", type=float, default=0.5,
                    help="B2 接触维持率下限（默认 0.5，即 >50%%）")
    ap.add_argument("--b2-contact-cutoff", type=float, default=4.5,
                    help="B2 接触判定距离（A，默认 4.5）")
    ap.add_argument("--b3-drift", type=float, default=5.0,
                    help="B3 质心漂移阈值（A，默认 5.0；文档 09 篇未定数值，按口袋尺度冻结）")
    ap.add_argument("--b3-max-frac", type=float, default=0.2,
                    help="B3 漂移超阈帧占比上限（默认 0.2，即 <20%%）")
    ap.add_argument("--b4-rmsd", type=float, default=4.0,
                    help="B4 位点骨架 RMSD 参考阈值（A，默认 4.0，文档 09 篇口径）")
    ap.add_argument("--b4-spike-frac", type=float, default=0.05,
                    help="B4 允许的瞬时尖峰帧占比上限（默认 0.05，用于区分孤立尖峰与长期超限）")
    ap.add_argument("--stride", type=int, default=1, help="轨迹抽帧步长（默认 1，逐帧分析）")
    ap.add_argument("--out", default=None, help="verdict JSON 输出路径（默认 verdict_<分子>.json）")
    args = ap.parse_args()

    labels = args.pose_label or ["pose%d" % (i + 1) for i in range(len(args.traj))]
    if len(labels) != len(args.traj):
        sys.exit("[错误] --pose-label 数量（%d）与 --traj 数量（%d）不一致" % (len(labels), len(args.traj)))
    for p in [args.topol] + args.traj:
        if not os.path.isfile(p):
            sys.exit("[错误] 文件不存在：%s" % p)

    poses, all_warn = [], []
    for label, traj in zip(labels, args.traj):
        print("[分析] %s <- %s ..." % (label, os.path.basename(traj)))
        res = analyze_pose(args.topol, traj, args)
        res["pose_label"] = label
        poses.append(res)
        all_warn.extend(res.pop("warnings"))
        print("  B1=%.2f B2=%s B3=%.2f B4=%.2f -> %s"
              % (res["B1_ligand_rmsd_max_A"], res["B2_contact_maintenance"],
                 res["B3_drift_over_threshold_frac"], res["B4_site_backbone_rmsd_max_A"],
                 "PASS" if res["overall"] else "FAIL"))

    # 文档 09 篇 2.5：任一 pose 四条全 PASS 则分子存活；全部失败才淘汰
    verdict = "SURVIVE" if any(p["overall"] for p in poses) else "ELIMINATE"
    out = {
        "molecule": args.molecule, "verdict": verdict,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": {"B1_rmsd_max_A": args.b1_rmsd, "B2_min_maintenance": args.b2_min,
                       "B2_contact_cutoff_A": args.b2_contact_cutoff,
                       "B3_drift_A": args.b3_drift, "B3_max_frac": args.b3_max_frac,
                       "B4_rmsd_A": args.b4_rmsd, "B4_spike_frac": args.b4_spike_frac},
        "rule": "任一 pose 四条判据全 PASS 则 SURVIVE，全部 pose 失败才 ELIMINATE（文档 09 篇 2.5 节）",
        "warnings": all_warn, "poses": poses,
    }
    out_path = args.out or "verdict_%s.json" % args.molecule
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[裁决] %s -> %s（verdict JSON：%s）" % (args.molecule, verdict, out_path))
    print("[提醒] 未部署 MD 的分子必须显式标注『未部署』，不能暗示做过（文档 09 篇验收标准）")


if __name__ == "__main__":
    main()
