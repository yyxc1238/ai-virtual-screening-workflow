#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01 蛋白结构预处理脚本（参考工作流文档 02 篇《靶点结构建模》）

功能：
1. 用 PDBFixer 做标准预处理：补缺失残基/侧链原子、按目标 pH 加氢、
   去除杂水与杂分子，输出可直接用于口袋识别/对接的干净受体 PDB
   （流程对应文档 02 篇 2.3 节与 3.2 节关键命令）；
2. 全程输出预处理日志（JSON），保证步骤可追溯（文档 02 篇"产出与验收标准"）；
3. 可选解析 AlphaFold 预测结构（PALA 同格式，pLDDT 存于 B-factor 列），
   输出 pLDDT 质量报告（均值、>70 占比、分档统计；分档阈值见文档 02 篇 2.2 节：
   >=90 高置信 / 70-90 中高 / 50-70 低 / <50 不可信）。

依赖：pdbfixer、openmm（推荐 conda install -c conda-forge pdbfixer openmm）
用法示例：
    python 01_prepare_structure.py --input receptor_raw.pdb --output receptor_clean.pdb
    python 01_prepare_structure.py --input af_model.pdb --plddt-only --report plddt_report.json
"""
import argparse
import json
import os
import sys


def check_deps():
    """检查 pdbfixer / openmm 是否可用，缺依赖时给出安装提示。"""
    try:
        from pdbfixer import PDBFixer  # noqa: F401
        from openmm.app import PDBFile  # noqa: F401
    except ImportError as e:
        sys.exit(
            "[错误] 缺少依赖：%s\n"
            "请先安装（推荐 conda-forge）：conda install -c conda-forge pdbfixer openmm\n"
            "或尝试 pip：pip install pdbfixer openmm" % e
        )


def prepare_structure(input_pdb, output_pdb, ph=7.0, keep_water=False):
    """标准预处理流程，逐步骤记录日志（命令顺序参考文档 02 篇 3.2 节）。"""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    log = {"input": os.path.basename(input_pdb), "ph": ph, "keep_water": keep_water}

    fixer = PDBFixer(filename=input_pdb)

    # 1) 补缺失残基（短 loop；文档提示：长缺失段 >10 aa 不可靠，需谨慎）
    fixer.findMissingResidues()
    log["missing_residue_segments"] = len(fixer.missingResidues)

    # 2) 找出缺失的侧链/末端原子并补齐
    fixer.findMissingAtoms()
    log["missing_sidechain_atoms"] = sum(len(v) for v in fixer.missingAtoms.values())
    log["missing_terminal_residues"] = len(fixer.missingTerminals)
    fixer.addMissingAtoms()

    # 3) 按目标 pH 加氢（文档 02 篇：默认 pH 7.0，注意组氨酸质子化态）
    fixer.addMissingHydrogens(pH=ph)

    # 4) 去杂：移除杂水/无关配体/离子（结晶水去留应由口袋分析决定，见文档 03 篇）
    fixer.removeHeterogens(keepWater=keep_water)
    log["heterogens_removed"] = not keep_water

    with open(output_pdb, "w", encoding="utf-8") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    log["output"] = os.path.basename(output_pdb)
    log["status"] = "done"
    return log


def plddt_report(af_pdb):
    """解析 AlphaFold 预测结构 CA 原子的 B-factor（即 pLDDT，0-100）。

    分档阈值参考工作流文档 02 篇 2.2 节：>=90 高；70-90 中高；50-70 低；<50 不可信。
    注意（文档 02 篇）：全局平均分高不代表口袋区域可靠，口袋残基区间需单独检查。
    """
    import numpy as np

    values = []
    with open(af_pdb, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                try:
                    values.append(float(line[60:66]))
                except ValueError:
                    continue
    if not values:
        sys.exit("[错误] 未解析到 CA 原子的 B-factor（pLDDT）。"
                 "请确认输入为 AlphaFold 预测结构（pLDDT 存于 B-factor 列）")
    arr = np.asarray(values, dtype=float)
    return {
        "file": os.path.basename(af_pdb),
        "n_residues": int(arr.size),
        "plddt_mean": round(float(arr.mean()), 2),
        "frac_gt_70": round(float((arr > 70).mean()), 4),
        "bucket_ge_90": round(float((arr >= 90).mean()), 4),
        "bucket_70_90": round(float(((arr >= 70) & (arr < 90)).mean()), 4),
        "bucket_50_70": round(float(((arr >= 50) & (arr < 70)).mean()), 4),
        "bucket_lt_50": round(float((arr < 50).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser(
        description="蛋白结构预处理（PDBFixer）+ AlphaFold pLDDT 质量报告，参考工作流文档 02 篇",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python 01_prepare_structure.py --input receptor_raw.pdb --output receptor_clean.pdb",
    )
    ap.add_argument("--input", required=True, help="输入 PDB（实验结构或 AlphaFold 预测结构）")
    ap.add_argument("--output", help="输出干净受体 PDB（默认在输入名后加 _clean.pdb）")
    ap.add_argument("--ph", type=float, default=7.0, help="加氢的目标 pH（默认 7.0，文档 02 篇口径）")
    ap.add_argument("--keep-water", action="store_true",
                    help="保留结晶水（默认去除；桥接关键水是否保留应结合口袋分析决定，见文档 03 篇）")
    ap.add_argument("--plddt-only", action="store_true",
                    help="只解析 pLDDT 并输出报告，不执行结构预处理（用于预测结构置信度评估）")
    ap.add_argument("--plddt-from", help="指定用于 pLDDT 解析的 AlphaFold 结构（默认用 --input 文件）")
    ap.add_argument("--report", help="日志/报告 JSON 输出路径（默认 <输出名>_prep_log.json）")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit("[错误] 输入文件不存在：%s" % args.input)

    stem = os.path.splitext(args.input)[0]

    # ---- 模式一：仅解析 pLDDT ----
    if args.plddt_only:
        rep = plddt_report(args.plddt_from or args.input)
        out_path = args.report or stem + "_plddt_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print("[pLDDT 报告] 均值=%.2f，>70 占比=%.1f%%（详见 %s）"
              % (rep["plddt_mean"], rep["frac_gt_70"] * 100, out_path))
        print("分档：>=90: %.1f%% | 70-90: %.1f%% | 50-70: %.1f%% | <50: %.1f%%"
              % (rep["bucket_ge_90"] * 100, rep["bucket_70_90"] * 100,
                 rep["bucket_50_70"] * 100, rep["bucket_lt_50"] * 100))
        return

    # ---- 模式二：完整预处理 ----
    check_deps()
    output_pdb = args.output or stem + "_clean.pdb"
    log = prepare_structure(args.input, output_pdb, ph=args.ph, keep_water=args.keep_water)

    # 可选：同时评估 pLDDT（预测结构建议保留该记录，文档 02 篇验收标准要求）
    if args.plddt_from:
        log["plddt"] = plddt_report(args.plddt_from)

    report_path = args.report or os.path.splitext(output_pdb)[0] + "_prep_log.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("[完成] 预处理输出：%s" % output_pdb)
    print("[日志] 缺失残基段=%d，补侧链原子=%d，pH=%.1f，去杂水杂分子=%s"
          % (log["missing_residue_segments"], log["missing_sidechain_atoms"],
             log["ph"], log["heterogens_removed"]))
    print("[日志文件] %s" % report_path)


if __name__ == "__main__":
    main()
