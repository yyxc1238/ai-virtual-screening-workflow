#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02 口袋识别与对接盒定义（参考工作流文档 03 篇《结合口袋识别与确认》）

功能（对应文档"三重证据交汇法"中的证据 1 无偏几何扫描 + 证据 2 共晶锚点）：
1. 封装 fpocket 命令行对受体表面做无偏口袋扫描，解析各口袋的体积与残基组成
   （跨工具核对以"残基集合"为准，体积仅作量级参考，见文档 03 篇）；
2. 给定参考配体（--ref-ligand，如共晶配体 PDB）时，统计与其接触（默认 4 A 内）
   的残基集合，自动筛选"接触参考配体残基最多的口袋"作为功能口袋候选；
3. 由口袋残基坐标推导对接盒：中心 = 口袋残基几何中心，尺寸 = 残基范围 +
   外延缓冲（--buffer，默认 4.0 A，文档口径 3-5 A），每边最小 20 A（文档口径
   通常 20-30 A），并按对接引擎网格间距（--grid-spacing，默认 0.375 A，Vina
   默认）对齐盒子尺寸；也支持 --box-from-ref 直接由参考配体质心/范围取盒。

依赖：fpocket（外部命令）+ numpy
  fpocket 安装：conda install -c conda-forge fpocket 或 apt install fpocket
用法示例：
    python 02_identify_pockets.py --receptor receptor_clean.pdb
    python 02_identify_pockets.py --receptor receptor_clean.pdb --ref-ligand ligand.pdb --box-from-ref
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("[错误] 缺少依赖 numpy。安装：pip install numpy")


def parse_pdb_atoms(path):
    """解析 PDB 原子坐标与残基标识，返回 (coords[N,3], residue_keys[N])。

    residue_key 格式为 "链:残基编号:残基名"，作为跨工具核对残基集合的统一标识。
    """
    coords, residues = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
            residues.append("%s:%s:%s" % (line[21].strip() or "-",
                                          line[22:26].strip(), line[17:20].strip()))
    if not coords:
        sys.exit("[错误] %s 中未解析到任何原子坐标" % path)
    return np.asarray(coords, dtype=np.float64), residues


def run_fpocket(receptor):
    """调用 fpocket 命令行（文档 03 篇 3.1 节：fpocket -f receptor_clean.pdb）。"""
    if shutil.which("fpocket") is None:
        sys.exit("[错误] 未找到 fpocket 命令。请先安装：\n"
                 "  conda install -c conda-forge fpocket\n"
                 "  或 apt install fpocket / 自行编译 https://github.com/Discngine/fpocket")
    print("[info] 运行 fpocket -f %s ..." % os.path.basename(receptor))
    r = subprocess.run(["fpocket", "-f", receptor], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("[错误] fpocket 运行失败：\n%s" % (r.stderr or r.stdout))
    stem = os.path.splitext(receptor)[0]
    out_dir = stem + "_out"
    if not os.path.isdir(out_dir):
        cands = sorted(glob.glob(stem + "*_out"))
        if not cands:
            sys.exit("[错误] 未找到 fpocket 输出目录（%s_out）" % stem)
        out_dir = cands[-1]
    return out_dir


def parse_pockets(out_dir):
    """解析 fpocket 输出：残基组成来自 pockets/pocketN_atm.pdb，体积来自 *_info.txt。"""
    volumes = {}
    for info in glob.glob(os.path.join(out_dir, "*_info.txt")):
        with open(info, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for m in re.finditer(r"Pocket\s+(\d+)\s*:.*?Volume\s*=\s*([0-9]+\.?[0-9]*(?:[eE][-+]?\d+)?)", text):
            volumes[int(m.group(1))] = float(m.group(2))
    pockets = []
    files = glob.glob(os.path.join(out_dir, "pockets", "pocket*_atm.pdb"))
    files.sort(key=lambda p: int(re.search(r"pocket(\d+)_atm", os.path.basename(p)).group(1)))
    for pf in files:
        pid = int(re.search(r"pocket(\d+)_atm", os.path.basename(pf)).group(1))
        coords, residues = parse_pdb_atoms(pf)
        pockets.append({
            "id": pid,
            "volume_A3": volumes.get(pid),
            "n_atoms": int(coords.shape[0]),
            "residues": sorted(set(residues)),
            "coords": coords,
        })
    if not pockets:
        sys.exit("[错误] fpocket 未输出口袋文件（pockets/pocket*_atm.pdb），请检查受体结构")
    return pockets


def contact_residue_keys(receptor, ligand_pdb, cutoff):
    """参考配体接触残基集合（文档 03 篇 3.2 节：与共晶配体的接触残基比对）。"""
    prot_coords, prot_res = parse_pdb_atoms(receptor)
    lig_coords, _ = parse_pdb_atoms(ligand_pdb)
    dmat = np.linalg.norm(prot_coords[:, None, :] - lig_coords[None, :, :], axis=2)
    close = dmat.min(axis=1) <= cutoff
    return set(r for r, c in zip(prot_res, close) if c)


def rank_pockets(pockets, ref_keys):
    """口袋排序：有参考配体时按"接触参考配体残基数"降序（文档 03 篇锚点优先），
    无参考配体时按体积降序（仅量级参考）。"""
    for p in pockets:
        p["residue_set"] = set(p["residues"])
        p["overlap_with_ref"] = len(p["residue_set"] & ref_keys) if ref_keys else None
    if ref_keys:
        pockets.sort(key=lambda p: (-p["overlap_with_ref"], -(p["volume_A3"] or 0.0)))
    else:
        pockets.sort(key=lambda p: -(p["volume_A3"] or 0.0))


def derive_box(coords, buffer, min_size, grid):
    """由原子坐标推导盒子：中心=几何中心；尺寸=范围+2*buffer，下限 min_size，
    并对齐到网格间距的整数倍（文档 03 篇 3.3 节）。"""
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    size = np.maximum(extent + 2.0 * buffer, min_size)
    if grid and grid > 0:
        size = np.round(size / grid) * grid
    return center, size


def main():
    ap = argparse.ArgumentParser(
        description="fpocket 口袋识别 + 参考配体锚点筛选 + 对接盒定义，参考工作流文档 03 篇",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python 02_identify_pockets.py --receptor receptor_clean.pdb --ref-ligand lig.pdb",
    )
    ap.add_argument("--receptor", required=True, help="预处理后的受体 PDB（01 脚本输出）")
    ap.add_argument("--ref-ligand", help="参考配体 PDB（如共晶配体），用于接触残基交叉验证")
    ap.add_argument("--contact-cutoff", type=float, default=4.0,
                    help="参考配体接触残基判定距离（A，默认 4.0）")
    ap.add_argument("--select-pocket", type=int, help="手动指定用于推导盒子的口袋编号（默认自动选最优）")
    ap.add_argument("--box-from-ref", action="store_true",
                    help="由参考配体质心/范围直接推导盒子（文档 03 篇：中心可取共晶配体质心）")
    ap.add_argument("--buffer", type=float, default=4.0,
                    help="盒子外延缓冲距离（A，默认 4.0；文档 03 篇口径 3-5 A）")
    ap.add_argument("--min-size", type=float, default=20.0,
                    help="盒子每边最小尺寸（A，默认 20.0；文档 03 篇口径通常 20-30 A）")
    ap.add_argument("--grid-spacing", type=float, default=0.375,
                    help="网格间距（A，默认 0.375 即 Vina 默认；<=0 表示不对齐）")
    ap.add_argument("--out", default="pockets_report.json", help="输出 JSON 报告路径")
    args = ap.parse_args()

    if not os.path.isfile(args.receptor):
        sys.exit("[错误] 受体文件不存在：%s" % args.receptor)
    if args.ref_ligand and not os.path.isfile(args.ref_ligand):
        sys.exit("[错误] 参考配体文件不存在：%s" % args.ref_ligand)

    out_dir = run_fpocket(args.receptor)
    pockets = parse_pockets(out_dir)

    ref_keys = set()
    if args.ref_ligand:
        ref_keys = contact_residue_keys(args.receptor, args.ref_ligand, args.contact_cutoff)
        print("[info] 参考配体接触残基 %d 个（cutoff=%.1f A）" % (len(ref_keys), args.contact_cutoff))

    rank_pockets(pockets, ref_keys)

    # ---- 打印口袋列表 ----
    print("\n%-6s %-14s %-8s %-12s %s" % ("ID", "Volume(A3)", "N_res", "OverlapRef", "口袋身份按残基集合核对，体积仅作量级参考（文档 03 篇）"))
    for p in pockets:
        ov = "-" if p["overlap_with_ref"] is None else str(p["overlap_with_ref"])
        print("%-6d %-14s %-8d %-12s" % (p["id"],
                                         ("%.1f" % p["volume_A3"]) if p["volume_A3"] else "n/a",
                                         len(p["residues"]), ov))

    # ---- 选定口袋 / 推导对接盒 ----
    if args.box_from_ref and args.ref_ligand:
        lig_coords, _ = parse_pdb_atoms(args.ref_ligand)
        center, size = derive_box(lig_coords, args.buffer, args.min_size, args.grid_spacing)
        box_source = "ref_ligand"
    else:
        if args.select_pocket:
            sel = next((p for p in pockets if p["id"] == args.select_pocket), None)
            if sel is None:
                sys.exit("[错误] 未找到口袋编号 %d（可用编号见上表）" % args.select_pocket)
        else:
            sel = pockets[0]
        center, size = derive_box(sel["coords"], args.buffer, args.min_size, args.grid_spacing)
        box_source = "pocket_%d" % sel["id"]

    print("\n[对接盒] center = %.3f %.3f %.3f | size = %.3f %.3f %.3f | source = %s"
          % (center[0], center[1], center[2], size[0], size[1], size[2], box_source))

    # ---- 输出 JSON 报告 ----
    report = {
        "receptor": os.path.basename(args.receptor),
        "ref_ligand": os.path.basename(args.ref_ligand) if args.ref_ligand else None,
        "contact_cutoff_A": args.contact_cutoff if args.ref_ligand else None,
        "pockets": [{
            "id": p["id"],
            "volume_A3": p["volume_A3"],
            "n_residues": len(p["residues"]),
            "overlap_with_ref": p["overlap_with_ref"],
            "residues": p["residues"],
        } for p in pockets],
        "box": {"center": [round(float(v), 3) for v in center],
                "size": [round(float(v), 3) for v in size],
                "buffer_A": args.buffer, "min_size_A": args.min_size,
                "grid_spacing_A": args.grid_spacing, "source": box_source},
        "note": "口袋身份按残基集合核对；盒子外延 3-5 A（默认 4.0），尺寸对齐 0.375 A 网格（文档 03 篇）",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[完成] 口袋档案与盒子参数已写入 %s" % args.out)


if __name__ == "__main__":
    main()
