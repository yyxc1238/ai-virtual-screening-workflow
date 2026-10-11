#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05 批量对接筛选脚本（参考工作流文档 05 篇《多线并行虚拟筛选》）

功能：
1. 用 Meeko 把 SDF/SMI 库分子转为 PDBQT，调用 AutoDock Vina（Python API）批量对接；
2. 支持多盒子并行筛选（--box 可多次指定，对应"多口袋策略"，文档 03/05 篇）；
3. 按 --gate1 阈值（默认 -7.0 kcal/mol，文档 05 篇"第 1 门对接分阈值"口径）
   出第一门名单，实现大库到千级的粗收敛；
4. 输出台账（输入/成功/失败/第一门通过数），保证每一级可对账（文档 05 篇第四节）。

公平性提醒（文档 05 篇 2.4 节铁律）：
- 同一条筛选线上所有分子必须使用完全相同的引擎、盒子与采样参数；
- 打分异常为正的分子按失败标记处理，不当作低分参与排序（文档第五节）；
- 不同引擎分数不可混排，本脚本榜单只与同引擎结果比较。

依赖：vina、meeko、rdkit、numpy
  安装：pip install vina>=1.2.3 meeko rdkit numpy
  受体 PDBQT 需提前制备（如 ADFR 套件 prepare_receptor 或 OpenBabel）。
用法示例：
    python 05_dock_screen.py --receptor receptor.pdbqt --input light_track.sdf \
        --box 12.5 8.0 3.0 24 24 24 --box 5.0 15.0 9.0 22 22 22 --out-dir docking_out
"""
import argparse
import csv
import json
import os
import re
import sys
import time

try:
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    sys.exit("[错误] 缺少依赖：%s\n安装：pip install numpy rdkit" % e)


def load_molecules(path, csv_smiles_col=None, csv_name_col=None):
    """读入库分子（sdf/smi/csv），返回 [(name, mol)]。"""
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
        import csv as _csv
        with open(path, encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        scol = csv_smiles_col or (list(rows[0].keys())[0] if rows else None)
        ncol = csv_name_col or (list(rows[0].keys())[1] if rows and len(rows[0]) > 1 else None)
        for i, row in enumerate(rows):
            m = Chem.MolFromSmiles(str(row.get(scol, "")).strip())
            if m is not None:
                mols.append((str(row.get(ncol, "mol%d" % i)) if ncol else "mol%d" % i, m))
    else:
        sys.exit("[错误] 不支持的库格式：%s（支持 sdf/smi/csv）" % ext)
    return mols


def prep_pdbqt(mol, embed_seed):
    """分子 -> 补氢 + 生成 3D 构象 -> Meeko -> PDBQT 字符串（缺氢会让对接全错）。"""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError:
        sys.exit("[错误] 缺少 meeko。安装：pip install meeko（Meeko 负责 SDF -> PDBQT）")
    m = Chem.AddHs(mol)  # 全氢校验思想：参数化必须在正确加氢的分子上进行（文档 04 篇 2.3）
    if m.GetNumConformers() == 0:
        if AllChem.EmbedMolecule(m, randomSeed=embed_seed) != 0:
            return None, "embed_failed"
    try:
        setups = MoleculePreparation().prepare(m)
        if not setups:
            return None, "prep_failed"
        s, ok, msg = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            return None, "write_failed: %s" % msg
        return s, None
    except Exception as exc:
        return None, "prep_error: %s" % exc


def main():
    ap = argparse.ArgumentParser(
        description="AutoDock Vina 批量对接 + 第一门（gate1）筛选，参考工作流文档 05 篇",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receptor", required=True, help="受体 PDBQT（提前制备并加氢）")
    ap.add_argument("--input", required=True, help="库文件（sdf/smi/csv，建议用 04 脚本输出）")
    ap.add_argument("--box", action="append", nargs=6, type=float, required=True,
                    metavar=("CX", "CY", "CZ", "SX", "SY", "SZ"),
                    help="对接盒（中心+尺寸，A；可多次指定实现多口袋并行）")
    ap.add_argument("--exhaustiveness", type=int, default=8,
                    help="Vina 采样彻底度（默认 8；筛选档全库统一，文档 05 篇 2.4）")
    ap.add_argument("--num-modes", type=int, default=9, help="保留构象模式数（默认 9）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（默认 0，全库统一）")
    ap.add_argument("--gate1", type=float, default=-7.0,
                    help="第一门对接分阈值 kcal/mol（默认 -7.0，文档 05 篇第一门口径；"
                         "文档未定死数值，应按靶点校准并在开跑前冻结）")
    ap.add_argument("--write-poses", action="store_true", help="保存每个分子的最优 pose（PDBQT）")
    ap.add_argument("--embed-seed", type=int, default=42, help="3D 构象生成随机种子（默认 42）")
    ap.add_argument("--csv-smiles-col", help="CSV 模式：SMILES 列名")
    ap.add_argument("--csv-name-col", help="CSV 模式：名称列名")
    ap.add_argument("--out-dir", default="docking_out", help="输出目录（默认 docking_out）")
    args = ap.parse_args()

    if not os.path.isfile(args.receptor):
        sys.exit("[错误] 受体文件不存在：%s" % args.receptor)
    if not os.path.isfile(args.input):
        sys.exit("[错误] 库文件不存在：%s" % args.input)
    try:
        from vina import Vina
    except ImportError:
        sys.exit("[错误] 缺少 vina 模块。安装：pip install vina>=1.2.3 "
                 "或 conda install -c conda-forge vina")

    mols = load_molecules(args.input, args.csv_smiles_col, args.csv_name_col)
    if not mols:
        sys.exit("[错误] 未从 %s 读到分子" % args.input)
    print("[info] 读入分子 %d 个，盒子 %d 个，exhaustiveness=%d，seed=%d"
          % (len(mols), len(args.box), args.exhaustiveness, args.seed))

    # 预制备 PDBQT（同一分子在多个盒子间复用，保证全库参数一致）
    prepared, prep_failures = [], {}
    for name, mol in mols:
        pdbqt, err = prep_pdbqt(mol, args.embed_seed)
        if err:
            prep_failures[name] = err
        else:
            prepared.append((name, Chem.MolToSmiles(Chem.RemoveHs(mol)), pdbqt))
    print("[info] 制备成功 %d，制备失败 %d" % (len(prepared), len(prep_failures)))

    poses_root = os.path.join(args.out_dir, "poses")
    if args.write_poses:
        os.makedirs(poses_root, exist_ok=True)

    ledgers, gate1_all = {}, []
    for bi, box in enumerate(args.box, start=1):
        cx, cy, cz, sx, sy, sz = box
        v = Vina(sf_name="vina", seed=args.seed, verbosity=1)
        v.set_receptor(args.receptor)
        v.compute_vina_maps(center=[cx, cy, cz], box_size=[sx, sy, sz])

        rows = []
        for name, smiles, pdbqt in prepared:
            try:
                v.set_ligand_from_string(pdbqt)
                v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.num_modes)
                score = float(v.energies()[0][0])
            except Exception as exc:
                rows.append([name, smiles, "", "dock_error: %s" % exc, "", ""])
                continue
            # 文档 05 篇：打分异常为正 = 失败/无效，须标记而非当作低分参与排序
            anomaly = score > 0
            gate1_pass = (not anomaly) and score <= args.gate1
            status = "anomaly_positive" if anomaly else "docked"
            if gate1_pass:
                gate1_all.append([name, "box%d" % bi, score])
            rows.append([name, smiles, "%.3f" % score, status, gate1_pass, anomaly])

        csv_path = os.path.join(args.out_dir, "docking_box%d.csv" % bi)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "smiles", "vina_score_kcal_mol", "status", "gate1_pass", "anomaly_positive"])
            w.writerows(rows)

        n_docked = sum(1 for r in rows if r[3] in ("docked", "anomaly_positive"))
        n_anom = sum(1 for r in rows if r[3] == "anomaly_positive")
        n_pass = sum(1 for r in rows if r[4] == True)  # noqa: E712
        ledgers["box%d" % bi] = {
            "center": [cx, cy, cz], "size": [sx, sy, sz],
            "n_input": len(prepared), "n_docked": n_docked,
            "n_prep_failed": len(prep_failures), "n_anomaly_positive": n_anom,
            "n_gate1_pass": n_pass, "gate1_threshold": args.gate1,
            "csv": os.path.basename(csv_path),
        }
        print("[box%d] 输入 %d | 对接成功 %d | 异常正分 %d | 第一门通过 %d"
              % (bi, len(prepared), n_docked, n_anom, n_pass))

        if args.write_poses:
            for name, _smi, pdbqt in prepared:
                try:
                    v.set_ligand_from_string(pdbqt)
                    v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.num_modes)
                    safe = re.sub(r"[^\w.-]+", "_", name)
                    v.write_poses(os.path.join(poses_root, "box%d_%s.pdbqt" % (bi, safe)),
                                  n_poses=1, overwrite=True)
                except Exception:
                    continue

    if gate1_all:
        gate1_all.sort(key=lambda r: r[2])
        with open(os.path.join(args.out_dir, "gate1_pass_list.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "box", "vina_score_kcal_mol"])
            w.writerows(gate1_all)

    ledgers["_global"] = {
        "engine": "AutoDock Vina (python API)", "exhaustiveness": args.exhaustiveness,
        "seed": args.seed, "n_molecules_loaded": len(mols),
        "prep_failures": prep_failures, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "第一门为高召回粗收敛；阈值应在开跑前冻结，全库参数一致（文档 05 篇）",
    }
    with open(os.path.join(args.out_dir, "docking_summary.json"), "w", encoding="utf-8") as f:
        json.dump(ledgers, f, ensure_ascii=False, indent=2)
    print("[输出] 台账与第一门名单已写入 %s" % args.out_dir)


if __name__ == "__main__":
    main()
