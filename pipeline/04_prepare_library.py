#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""04 分子库清洗与分轨脚本（参考工作流文档 04 篇《分子库构建与分轨》）

流程（对应文档第三节操作步骤）：
  原始库（SMI/CSV/SDF，可多输入）
    -> 去盐（保留最大有机片段）
    -> 标准化（统一官能团表示）
    -> 质子化校正（pH 7.0 口径）
    -> 垃圾过滤（MW 超限、含金属等，文档 2.2 节）
    -> 全氢校验（文档 2.3 节关键质控：断言通过才放行，避免缺氢导致相互作用失真）
    -> 去重（canonical SMILES + InChIKey 双重，文档 2.2 节）
    -> 按重原子数分轨（默认阈值 45，约 MW 500 以下；文档 2.4 节：
       轻线=类药小分子，重线=大柔性分子，采样强度与复杂度匹配）
  输出：轻/重两轨 SDF + SMI + 清洗日志 CSV + 汇总 JSON

依赖：rdkit（安装：pip install rdkit 或 conda install -c conda-forge rdkit）
用法示例：
    python 04_prepare_library.py --input library.smi --input extra.sdf --out-dir prepared_library
    python 04_prepare_library.py --input lib.csv --csv-smiles-col smiles --csv-name-col name
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, rdMolStandardize
except ImportError as e:
    sys.exit("[错误] 缺少依赖 rdkit：%s\n安装：pip install rdkit 或 conda install -c conda-forge rdkit" % e)

# 文档 04 篇 2.2 节：过滤含金属/放射性元素的"明显垃圾"结构
METAL_SYMBOLS = frozenset((
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Al", "Ga", "In", "Sn", "Pb", "Bi", "Po", "Ra",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ac", "Th", "Pa", "U", "Np", "Pu",
))


def clean_mol(mol, periodic_table, min_mw, max_mw):
    """按文档 04 篇执行单分子清洗，返回 (mol, formula, reason)。

    reason 为 None 表示通过全部清洗与校验；否则为剔除原因分类。
    """
    if mol is None:
        return None, "", "parse_failed"
    try:
        # 1) 去盐：保留最大有机片段（文档 2.2 节第 1 条）
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        # 2) 标准化：统一官能团表示（硝基、磺酸等，文档 2.2 节第 3 条）
        mol = rdMolStandardize.Normalizer().normalize(mol)
        # 3) 质子化校正：pH 7.0 口径（文档 2.2 节第 5 条）
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "", "standardize_failed"

    # 4) 垃圾过滤：分子量范围（文档 2.2 节第 6 条及示例代码 MW<100 / >1000）
    mw = Descriptors.MolWt(mol)
    if mw < min_mw or mw > max_mw:
        return None, "", "mw_out_of_range"
    # 5) 含金属过滤（文档 2.2 节第 6 条）
    if any(periodic_table.GetElementSymbol(a.GetAtomicNum()) in METAL_SYMBOLS
           for a in mol.GetAtoms()):
        return None, "", "contains_metal"

    # 6) 全氢校验（文档 2.3 节）：加氢后化学式与 canonical SMILES 回读加氢化学式必须一致，
    #    防止 SMILES 来源问题导致缺氢分子进入参数化环节
    try:
        canon = Chem.MolToSmiles(mol)
        mol_rt = Chem.MolFromSmiles(canon)
        if mol_rt is None:
            return None, "", "full_hydrogen_check_failed"
        f1 = rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol))
        f2 = rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol_rt))
        if f1 != f2:
            return None, "", "full_hydrogen_check_failed"
    except Exception:
        return None, "", "full_hydrogen_check_failed"
    return mol, f1, None


def load_input(path, csv_smiles_col=None, csv_name_col=None):
    """读取原始库（sdf/smi/csv），产出 (来源标记, name, SMILES 或 None, mol 或 None)。"""
    ext = os.path.splitext(path)[1].lower()
    items = []
    if ext in (".sdf", ".sd"):
        for i, m in enumerate(Chem.SDMolSupplier(path)):
            name = m.GetProp("_Name") if (m is not None and m.HasProp("_Name")) else "mol%d" % i
            items.append((os.path.basename(path), name, None, m))
    elif ext in (".smi", ".smiles"):
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if not parts:
                continue
            m = Chem.MolFromSmiles(parts[0])
            items.append((os.path.basename(path), parts[1] if len(parts) > 1 else "mol%d" % len(items),
                          parts[0], m))
    elif ext == ".csv":
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        scol = csv_smiles_col or (list(rows[0].keys())[0] if rows else None)
        ncol = csv_name_col or (list(rows[0].keys())[1] if rows and len(rows[0]) > 1 else None)
        for i, row in enumerate(rows):
            m = Chem.MolFromSmiles(str(row.get(scol, "")).strip())
            items.append((os.path.basename(path), str(row.get(ncol, "mol%d" % i)) if ncol else "mol%d" % i,
                          str(row.get(scol, "")), m))
    else:
        sys.exit("[错误] 不支持的输入格式：%s（支持 sdf/smi/csv）" % ext)
    return items


def main():
    ap = argparse.ArgumentParser(
        description="分子库清洗（去盐/标准化/去重/全氢校验）与按重原子数分轨，参考工作流文档 04 篇",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True, help="输入库文件（sdf/smi/csv，可多次指定）")
    ap.add_argument("--out-dir", default="prepared_library", help="输出目录（默认 prepared_library）")
    ap.add_argument("--heavy-atom-threshold", type=int, default=45,
                    help="重原子数分轨阈值（默认 45，约 MW 500 以下；文档 04 篇 2.4 节）")
    ap.add_argument("--min-mw", type=float, default=100.0,
                    help="分子量下限（默认 100，低于则剔除；文档 04 篇 2.2 节）")
    ap.add_argument("--max-mw", type=float, default=1000.0,
                    help="分子量上限（默认 1000，超出则剔除；文档 04 篇示例代码口径）")
    ap.add_argument("--csv-smiles-col", help="CSV 模式：SMILES 列名")
    ap.add_argument("--csv-name-col", help="CSV 模式：名称列名")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    periodic_table = Chem.GetPeriodicTable()
    writer_light = Chem.SDWriter(os.path.join(args.out_dir, "light_track.sdf"))
    writer_heavy = Chem.SDWriter(os.path.join(args.out_dir, "heavy_track.sdf"))
    smi_light = open(os.path.join(args.out_dir, "light_track.smi"), "w", encoding="utf-8")
    smi_heavy = open(os.path.join(args.out_dir, "heavy_track.smi"), "w", encoding="utf-8")

    seen_canon, seen_inchikey = set(), set()   # 双重去重键（文档 2.2 节第 2 条）
    log_rows, reason_counter = [], Counter()
    n_in = n_light = n_heavy = 0

    for path in args.input:
        if not os.path.isfile(path):
            sys.exit("[错误] 输入文件不存在：%s" % path)
        for source, name, smi_in, mol in load_input(path, args.csv_smiles_col, args.csv_name_col):
            n_in += 1
            mol, formula, reason = clean_mol(mol, periodic_table, args.min_mw, args.max_mw)
            if reason:
                reason_counter[reason] += 1
                log_rows.append([source, name, smi_in or "", "", "rejected", "-", reason])
                continue

            canon = Chem.MolToSmiles(mol)
            try:
                inchikey = Chem.MolToInchiKey(mol) or ""
            except Exception:
                inchikey = ""
            if canon in seen_canon or (inchikey and inchikey in seen_inchikey):
                reason_counter["duplicate"] += 1
                log_rows.append([source, name, smi_in or canon, canon, "rejected", "-", "duplicate"])
                continue
            seen_canon.add(canon)
            if inchikey:
                seen_inchikey.add(inchikey)

            n_heavy_atoms = mol.GetNumHeavyAtoms()
            track = "light" if n_heavy_atoms <= args.heavy_atom_threshold else "heavy"
            mol.SetProp("_Name", name)
            mol.SetProp("canonical_smiles", canon)
            mol.SetProp("InChIKey", inchikey)
            mol.SetProp("heavy_atoms", str(n_heavy_atoms))
            mol.SetProp("formula", formula)
            if track == "light":
                writer_light.write(mol)
                smi_light.write("%s %s\n" % (canon, name))
                n_light += 1
            else:
                writer_heavy.write(mol)
                smi_heavy.write("%s %s\n" % (canon, name))
                n_heavy += 1
            log_rows.append([source, name, smi_in or canon, canon, "kept", track, ""])

    writer_light.close()
    writer_heavy.close()
    smi_light.close()
    smi_heavy.close()

    with open(os.path.join(args.out_dir, "cleaning_log.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "name", "smiles_in", "smiles_out", "status", "track", "reason"])
        w.writerows(log_rows)

    summary = {
        "n_input": n_in,
        "n_kept": n_light + n_heavy,
        "n_light_track": n_light,
        "n_heavy_track": n_heavy,
        "heavy_atom_threshold": args.heavy_atom_threshold,
        "rejections_by_reason": dict(reason_counter),
        "note": "数字口径区分：原始下载数 / 清洗后入库数 / 进入正式筛选数（文档 04 篇第六节）",
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[台账] 输入 %d | 保留 %d（轻线 %d / 重线 %d）" % (n_in, n_light + n_heavy, n_light, n_heavy))
    for reason, cnt in reason_counter.most_common():
        print("  剔除原因 %-28s %d" % (reason, cnt))
    print("[输出] 轻线: light_track.sdf / .smi | 重线: heavy_track.sdf / .smi（目录 %s）" % args.out_dir)


if __name__ == "__main__":
    main()
