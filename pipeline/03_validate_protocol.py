#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""03 筛选前协议验证脚本（参考工作流文档 05 篇 2.6 节与 3.0 节）

文档核心要求："大规模开跑前，先验证对接协议本身是否可信——验证不过，不许进入全库筛选。"
两种验证模式（按手头条件二选一）：
1. redock（有共晶配体，首选）：把共晶配体用既定协议重新对回口袋，
   预测最优构象 vs 实验构象 RMSD <= 2 A 判 PASS（文档 05 篇经验阈值）；
2. anchor（无共晶）：把已知阳性对照分子当普通分子投入对接，
   报告其分数与排名分位——较弱但可操作的替代证据，不等同于严格 redocking。
结果输出 JSON 验证报告存档，通过后方可进入全库筛选（05 篇 3.0 节）。

依赖：vina、meeko、rdkit、numpy
  安装：pip install vina>=1.2.3 meeko rdkit numpy
用法示例：
    python 03_validate_protocol.py --mode redock --receptor receptor.pdbqt \
        --ligand lig_dock.pdbqt --ref lig_crystal.pdb --center 1 2 3 --size 22 22 22
    python 03_validate_protocol.py --mode anchor --receptor receptor.pdbqt \
        --library known_actives.smi --anchor-name anchor01 \
        --center 1 2 3 --size 22 22 22
"""
import argparse
import json
import os
import sys
import time

try:
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    sys.exit("[错误] 缺少依赖：%s\n安装：pip install numpy rdkit" % e)


def parse_coords(path_or_lines, is_string=False):
    """从 PDB/PDBQT 解析原子坐标与是否氢的标记，返回 (coords[N,3], is_H[N])。"""
    lines = path_or_lines.splitlines() if is_string else open(
        path_or_lines, encoding="utf-8", errors="ignore").read().splitlines()
    coords, is_h = [], []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
        t = line[77:79].strip().upper()  # PDBQT 的 AutoDock 原子类型（HD/H/D 为氢）
        nm = line[12:16].strip().upper()
        is_h.append(t in ("H", "HD", "D") or (not t and nm.startswith(("H", "D")) and not nm.startswith("HG")))
    if not coords:
        sys.exit("[错误] 未解析到原子坐标（检查输入是否为 PDB/PDBQT 格式）")
    return np.asarray(coords, dtype=np.float64), np.asarray(is_h, dtype=bool)


def first_pose_coords(pdbqt_string):
    """从 vina 多 MODEL 输出中取第一个（最优）pose 的坐标。"""
    coords, is_h = [], []
    for line in pdbqt_string.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        t = line[77:79].strip().upper()
        nm = line[12:16].strip().upper()
        is_h.append(t in ("H", "HD", "D") or (not t and nm.startswith(("H", "D")) and not nm.startswith("HG")))
    return np.asarray(coords, dtype=np.float64), np.asarray(is_h, dtype=bool)


def calc_rmsd(a, b):
    """重原子 RMSD（要求两边原子一一对应）。"""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def make_vina(receptor, seed):
    try:
        from vina import Vina
    except ImportError:
        sys.exit("[错误] 缺少 vina 模块。安装：pip install vina>=1.2.3 "
                 "或 conda install -c conda-forge vina")
    v = Vina(sf_name="vina", seed=seed, verbosity=1)
    v.set_receptor(receptor)
    return v


def prep_ligand_pdbqt(mol, embed_seed=42):
    """RDKit 分子 -> Meeko -> PDBQT 字符串（补氢、无构象时先生成 3D 构象）。"""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError:
        sys.exit("[错误] 缺少 meeko。安装：pip install meeko")
    m = Chem.AddHs(mol)
    if m.GetNumConformers() == 0:
        if AllChem.EmbedMolecule(m, randomSeed=embed_seed) != 0:
            return None, "embed_failed"
    setups = MoleculePreparation().prepare(m)
    if not setups:
        return None, "prep_failed"
    s, ok, msg = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        return None, "write_failed: %s" % msg
    return s, None


def load_molecules(path, csv_smiles_col=None, csv_name_col=None):
    """读入库分子（smi/sdf/csv），返回 [(name, mol)]。"""
    ext = os.path.splitext(path)[1].lower()
    mols = []
    if ext in (".sdf", ".sd"):
        for i, m in enumerate(Chem.SDMolSupplier(path)):
            if m is not None:
                mols.append((m.GetProp("_Name") if m.HasProp("_Name") else "mol%d" % i, m))
    elif ext in (".smi", ".smiles"):
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if not parts:
                continue
            m = Chem.MolFromSmiles(parts[0])
            if m is not None:
                mols.append((parts[1] if len(parts) > 1 else "mol%d" % len(mols), m))
    elif ext == ".csv":
        import csv
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        scol = csv_smiles_col or (list(rows[0].keys())[0] if rows else None)
        ncol = csv_name_col or (list(rows[0].keys())[1] if rows and len(rows[0]) > 1 else None)
        for i, row in enumerate(rows):
            m = Chem.MolFromSmiles(str(row.get(scol, "")).strip())
            if m is not None:
                mols.append((str(row.get(ncol, "mol%d" % i)) if ncol else "mol%d" % i, m))
    else:
        sys.exit("[错误] 不支持的库格式：%s（支持 smi/sdf/csv）" % ext)
    return mols


def run_redock(args):
    """自对接验证：预测最优构象 vs 实验构象 RMSD <= 2 A 判 PASS（文档 05 篇 2.6）。"""
    ref_coords, ref_h = parse_coords(args.ref)
    if args.heavy_only:
        ref_coords = ref_coords[~ref_h]

    if args.auto_box:
        center = ref_coords.mean(axis=0)
        size = np.maximum(ref_coords.max(axis=0) - ref_coords.min(axis=0) + 2 * args.buffer, 20.0)
    else:
        center, size = np.array(args.center, dtype=float), np.array(args.size, dtype=float)

    with open(args.ligand, encoding="utf-8") as f:
        lig_pdbqt = f.read()
    v = make_vina(args.receptor, args.seed)
    v.compute_vina_maps(center=list(center), box_size=list(size))
    v.set_ligand_from_string(lig_pdbqt)
    v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.n_poses)
    energy = float(v.energies()[0][0])

    pose_coords, pose_h = first_pose_coords(v.poses())
    if args.heavy_only:
        pose_coords = pose_coords[~pose_h]
    if pose_coords.shape[0] != ref_coords.shape[0]:
        sys.exit("[错误] 原子数不一致（pose %d vs 参考 %d）。redocking RMSD 要求两边原子一一对应："
                 "请用同一分子、同一质子化状态制备配体与参考构象，或检查 --heavy-only 设置"
                 % (pose_coords.shape[0], ref_coords.shape[0]))
    rmsd = calc_rmsd(pose_coords, ref_coords)
    verdict = "PASS" if rmsd <= args.rmsd_threshold else "FAIL"
    return {"mode": "redock", "best_energy_kcal_mol": round(energy, 3),
            "rmsd_A": round(rmsd, 3), "rmsd_threshold_A": args.rmsd_threshold,
            "verdict": verdict,
            "note": "RMSD <= %.1f A 说明协议能还原已知结合模式（文档 05 篇 2.6 节）" % args.rmsd_threshold}


def run_anchor(args):
    """阳性锚点验证：报告锚点分子分数与排名分位（文档 05 篇 2.6 节第三种情形）。"""
    if not args.anchor_name and not args.anchor_smiles:
        sys.exit("[错误] anchor 模式需提供 --anchor-name 或 --anchor-smiles 以识别阳性分子")
    mols = load_molecules(args.library, args.csv_smiles_col, args.csv_name_col)
    if not mols:
        sys.exit("[错误] 未从 %s 读到分子" % args.library)
    anchor_canon = Chem.MolToSmiles(Chem.MolFromSmiles(args.anchor_smiles)) if args.anchor_smiles else None

    v = make_vina(args.receptor, args.seed)
    v.compute_vina_maps(center=list(args.center), box_size=list(args.size))
    scores, failures, anchor_score = [], {}, None
    for name, mol in mols:
        pdbqt, err = prep_ligand_pdbqt(mol, args.embed_seed)
        if err:
            failures[name] = err
            continue
        try:
            v.set_ligand_from_string(pdbqt)
            v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.n_poses)
            e = float(v.energies()[0][0])
        except Exception as exc:
            failures[name] = "dock_error: %s" % exc
            continue
        scores.append((e, name))
        if (args.anchor_name and name == args.anchor_name) or \
           (anchor_canon and Chem.MolToSmiles(mol) == anchor_canon):
            anchor_score = e
    if anchor_score is None:
        sys.exit("[错误] 未在库中找到锚点分子（检查 --anchor-name / --anchor-smiles）")
    better = sum(1 for e, _ in scores if e < anchor_score)
    n = len(scores)
    rank = better + 1
    return {"mode": "anchor", "n_docked": n, "n_failed": len(failures), "failures": failures,
            "anchor_score_kcal_mol": round(anchor_score, 3), "anchor_rank": rank,
            "anchor_percentile": round(rank / n, 4),
            "best_score_kcal_mol": round(min(e for e, _ in scores), 3),
            "note": "锚点验证是较弱证据，不等同于严格 redocking（文档 05 篇 2.6 节）；"
                    "名次分位仅供参考，需结合接触残基合理性人工判断"}


def main():
    ap = argparse.ArgumentParser(
        description="筛选前协议验证（redocking / 阳性锚点），参考工作流文档 05 篇 2.6 与 3.0 节",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["redock", "anchor"], required=True, help="验证模式")
    ap.add_argument("--receptor", required=True, help="受体 PDBQT（需提前用 ADFR/prepare_receptor 等工具制备）")
    ap.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"), help="盒子中心（anchor 模式必填）")
    ap.add_argument("--size", nargs=3, type=float, metavar=("SX", "SY", "SZ"), help="盒子尺寸（anchor 模式必填）")
    # redock 参数
    ap.add_argument("--ligand", help="redock：用于对接的共晶配体 PDBQT")
    ap.add_argument("--ref", help="redock：共晶配体实验构象（PDB/PDBQT，原子顺序需与 --ligand 一致）")
    ap.add_argument("--auto-box", action="store_true", help="redock：由参考配体坐标自动取盒（外延 --buffer）")
    ap.add_argument("--buffer", type=float, default=4.0, help="auto-box 外延缓冲（A，默认 4.0，文档 03 篇口径）")
    ap.add_argument("--rmsd-threshold", type=float, default=2.0,
                    help="redocking RMSD 通过阈值（A，默认 2.0，文档 05 篇口径）")
    ap.add_argument("--heavy-only", dest="heavy_only", action="store_true", default=True,
                    help="RMSD 只用重原子（默认开启）")
    ap.add_argument("--all-atom-rmsd", dest="heavy_only", action="store_false", help="RMSD 使用全部原子")
    # anchor 参数
    ap.add_argument("--library", help="anchor：含阳性分子的库文件（smi/sdf/csv）")
    ap.add_argument("--anchor-name", help="anchor：阳性分子在库中的名称")
    ap.add_argument("--anchor-smiles", help="anchor：阳性分子的 SMILES（用于 canonical 匹配）")
    ap.add_argument("--csv-smiles-col", help="CSV 模式：SMILES 列名")
    ap.add_argument("--csv-name-col", help="CSV 模式：名称列名")
    # 通用对接参数（同一协议内全库一致，见文档 05 篇 2.4 公平性铁律）
    ap.add_argument("--exhaustiveness", type=int, default=16, help="Vina 采样彻底度（默认 16）")
    ap.add_argument("--n-poses", type=int, default=5, help="保留构象数（默认 5）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    ap.add_argument("--embed-seed", type=int, default=42, help="3D 构象生成随机种子（默认 42）")
    ap.add_argument("--out", default="protocol_validation.json", help="JSON 验证报告输出路径")
    args = ap.parse_args()

    if not os.path.isfile(args.receptor):
        sys.exit("[错误] 受体文件不存在：%s" % args.receptor)

    if args.mode == "redock":
        if not (args.ligand and args.ref):
            sys.exit("[错误] redock 模式需要 --ligand 与 --ref")
        result = run_redock(args)
    else:
        if not (args.center and args.size):
            sys.exit("[错误] anchor 模式需要 --center 与 --size（可由 02 脚本的盒子参数获得）")
        result = run_anchor(args)

    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "receptor": os.path.basename(args.receptor),
              "engine": "AutoDock Vina (python API)",
              "params": {"exhaustiveness": args.exhaustiveness, "n_poses": args.n_poses,
                         "seed": args.seed,
                         "center": list(args.center) if args.center else None,
                         "size": list(args.size) if args.size else None},
              "result": result}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if result["mode"] == "redock":
        print("[结果] RMSD = %.3f A（阈值 %.1f）-> %s" % (result["rmsd_A"], result["rmsd_threshold_A"], result["verdict"]))
    else:
        print("[结果] 锚点分数 %.3f kcal/mol，排名 %d/%d（分位 %.1f%%）"
              % (result["anchor_score_kcal_mol"], result["anchor_rank"],
                 result["n_docked"], result["anchor_percentile"] * 100))
    print("[报告] 已写入 %s（验证通过并留档后方可全库开跑，文档 05 篇 3.0 节）" % args.out)


if __name__ == "__main__":
    main()
