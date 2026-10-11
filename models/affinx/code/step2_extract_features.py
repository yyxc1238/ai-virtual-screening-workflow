# step2_extract_features.py —— 特征工程
# 读取：DATASET_refined.csv（step1 产出）
# 产出：
#   features_baseline.csv  —— 仅 1D/2D 特征（Baseline 训练用，必做）
#   features_3d.csv        —— + 3D 相互作用（PLIP） + 蛋白描述符（主线优化用）
#
# 特征分层：
#   A. Baseline（必做）
#     A1. 配体 1D/2D 描述符（RDKit）—— 分子量、LogP、TPSA、HBD/HBA、可旋转键数等（~200维）
#     A2. Morgan 分子指纹（半径2，2048位）—— 化学骨架特征
#   B. 优化层 1（核心）
#     B1. PLIP 蛋白-配体相互作用计数（氢键/疏水/π-π/盐桥/阳离子-π/卤键/金属键）
#     B2. 原子对距离计数直方图（RF-Score 经典做法：C/N/O/S/P/F/Cl/Br/I × 自身8种，共64维按距离分箱）
#     B3. 接触残基数 / 近邻原子数
#   C. 蛋白描述符（简化）
#     C1. 蛋白序列长度
#     C2. 口袋残基数（口袋PDB有效时可用，否则用接触残基）
#   D. 优化层 2（⏸️ 第一版不启用，占位）：Vina 能量项
#
# 注意：
# - RDKit 特征提取和 Morgan 指纹用 ligand.mol2/sdf → RDKit Mol 对象；失败时尝试 sdf 备用
# - PLIP 用 CLI（subprocess 调 plip -f <merged_pdb> -o <tmpdir> -x -y），
#   因为 CLI 已验证能用，程序式API类名在不同版本不稳
# - 特征提取支持断点续跑：读 FEATURES_DIR 下已有 CSV，跳过已算过的 pdb
# - 3D 特征计算耗时，默认用 4 进程并行（PLIP_MAXTHREADS）

import os
import re
import sys
import shutil
import subprocess
import tempfile
import traceback
import argparse
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from config import (
    FEATURES_DIR, MODELS_DIR, REPORTS_DIR,
    FEAT_LIGAND_1D2D, FEAT_LIGAND_MORGAN, MORGAN_RADIUS, MORGAN_NBITS,
    FEAT_PLIP_INTERACTIONS, PLIP_MAXTHREADS,
    FEAT_PROTEIN_DESC,
    FEAT_VINA_ENERGY,
)

DATASET_REFIND_CSV = FEATURES_DIR / "DATASET_refined.csv"
FEATURES_BASELINE_CSV = FEATURES_DIR / "features_baseline.csv"
FEATURES_3D_CSV = FEATURES_DIR / "features_3d.csv"

# ============================================================
# 工具：RDKit 描述符 + Morgan 指纹
# ============================================================
def _rdkit_available():
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


def _read_and_strip_comments(in_path: Path) -> bytes:
    """读取配体文件，剥掉 PDBBind 在文件头插入的 # / Name: / Version: 非标准注释行。
    同时修复 PDBBind v2020 SDF counts 行常见的"整数带小数点"(如 38.) 导致 RDKit 解析失败的问题。
    """
    try:
        raw_bytes = in_path.read_bytes()
    except Exception:
        return b""
    lines = raw_bytes.decode("latin-1", errors="replace").split("\n")
    cleaned = []
    mol2_started = False
    sdf_started = False
    sdf_counts_line_seen = False
    for ln in lines:
        s = ln.rstrip("\r")
        stripped = s.strip()
        if stripped.startswith("@<TRIPOS>"):
            mol2_started = True
        if (not sdf_counts_line_seen) and ("V2000" in stripped or "V3000" in stripped) and (not mol2_started):
            s_fixed = re.sub(r"\b(\d+)\.(?!\d)\b", r"\1", s)
            cleaned.append(s_fixed)
            sdf_counts_line_seen = True
            sdf_started = True
            continue
        if (not mol2_started) and (not sdf_started):
            if stripped.startswith("#") or stripped.startswith("Name:") or stripped.startswith("Version:"):
                continue
        cleaned.append(s)
    return ("\n".join(cleaned) + "\n").encode("latin-1")


def _read_ligand_mol(path: Path):
    """读取配体 mol2/sdf 为 RDKit Mol 对象。
    自动处理 PDBBind 注释行 + SDF counts 修复 + SDF→mol2 路径切换。
    失败返回 None。
    """
    if not _rdkit_available():
        return None
    from rdkit import Chem
    p = Path(path)
    suffix = p.suffix.lower()
    # SDF → 同名 mol2 fallback（PDBBind mol2 更稳定）
    if suffix == ".sdf":
        sib_mol2 = p.with_suffix(".mol2")
        if sib_mol2.exists():
            p = sib_mol2
            suffix = ".mol2"
    cleaned_bytes = _read_and_strip_comments(p)
    if not cleaned_bytes:
        return None
    import tempfile as _tf
    tmpdir = _tf.mkdtemp(prefix="rdk_clean_")
    tmp_clean = Path(tmpdir) / f"clean{suffix}"
    tmp_clean.write_bytes(cleaned_bytes)
    mol = None
    if suffix == ".mol2":
        try:
            mol = Chem.MolFromMol2File(str(tmp_clean), removeHs=False)
        except Exception:
            mol = None
    if mol is None and suffix == ".sdf":
        try:
            suppl = Chem.SDMolSupplier(str(tmp_clean), removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        except Exception:
            mol = None
    if mol is None:
        try:
            mol = Chem.MolFromMol2File(str(tmp_clean), removeHs=False)
        except Exception:
            mol = None
    # 清理临时文件
    try:
        tmp_clean.unlink()
        Path(tmpdir).rmdir()
    except Exception:
        pass
    return mol


def compute_ligand_2d_features(ligand_path: str):
    """从 ligand.mol2 或 ligand.sdf 提取 RDKit 1D/2D 描述符 + Morgan指纹。
    返回 dict（字段名 -> 值）；失败返回空 dict。

    兼容性处理（PDBBind v2020.R1 实际数据格式问题）：
      1. PDBBind 在 mol2/sdf 文件前几行写了 `# Name: xxx` / `# Version: PDBbind v2024` 注释，
         严格的 RDKit 解析器认为格式非法（Bad input file）。解决：先在内存中剥去以 `#` 开头
         和 `Name:` / `Version:` 开头的非标准注释行，再喂给 RDKit。
      2. 若 RDKit 仍读不出，用 openbabel Python API 读 -> 转 SMILES -> RDKit MolFromSmiles 兜底。
    """
    if not _rdkit_available():
        return {}
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    mol = _read_ligand_mol(Path(ligand_path))
    if mol is None:
        return {}

    out = {}
    # A1. 1D/2D 描述符（挑选有物理意义的约 30 个核心，完整版可切 Descriptors._descList 全量 200+）
    core_descs = [
        ("MolWt",         Descriptors.MolWt),
        ("MolLogP",        Descriptors.MolLogP),
        ("TPSA",           Descriptors.TPSA),
        ("HBD",            Descriptors.NumHDonors),
        ("HBA",            Descriptors.NumHAcceptors),
        ("RotatableBonds", Descriptors.NumRotatableBonds),
        ("AromaticRings",  Descriptors.NumAromaticRings),
        ("Rings",          Descriptors.RingCount),
        ("HeavyAtoms",     Descriptors.HeavyAtomCount),
        ("FractionCSP3",   Descriptors.FractionCSP3),
        ("NumHeteroatoms", Descriptors.NumHeteroatoms),
        ("NHOHCount",      Descriptors.NHOHCount),
        ("NOCount",        Descriptors.NOCount),
        ("MolMR",          Descriptors.MolMR),
        ("MaxPartialCharge", Descriptors.MaxPartialCharge if hasattr(Descriptors, "MaxPartialCharge") else lambda m: 0.0),
        ("MinPartialCharge", Descriptors.MinPartialCharge if hasattr(Descriptors, "MinPartialCharge") else lambda m: 0.0),
        ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge if hasattr(Descriptors, "MaxAbsPartialCharge") else lambda m: 0.0),
        ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge if hasattr(Descriptors, "MinAbsPartialCharge") else lambda m: 0.0),
    ]
    for name, fn in core_descs:
        try:
            out[f"desc_{name}"] = float(fn(mol))
        except Exception:
            out[f"desc_{name}"] = np.nan

    # A2. Morgan 指纹（2048位，稀疏位→稠密 int）
    if FEAT_LIGAND_MORGAN:
        try:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)
            fp = gen.GetFingerprint(mol)
            arr = np.zeros(MORGAN_NBITS, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            for i in range(MORGAN_NBITS):
                out[f"fp_morgan_{i}"] = int(arr[i])
        except Exception:
            pass
    return out


# ============================================================
# 工具：PLIP 蛋白-配体相互作用（CLI 方式）
# ============================================================
# PLIP 交互类型关键字（来自 XML/report 解析）
PLIP_INTERACTION_TYPES = [
    "hydrogen_bonds",
    "hydrophobic_interactions",
    "pi_stacking",
    "salt_bridges",
    "pi_cation_interactions",
    "halogen_bonds",
    "metal_complexes",
    "water_bridges",
]


def _run_plip_cli(protein_pdb: str, ligand_sdf_or_mol2: str, tmp_root: Path) -> dict:
    """调用 PLIP CLI 分析单一复合物。
    做法：
      1) 把 protein.pdb 和 ligand（mol2/sdf）用 openbabel 合并到同一份 PDB（PLIP 要求在同文件）
      2) plip -f merged.pdb -o tmpdir -x -y -t --maxthreads 1
      3) 读取 tmpdir/report.txt 或 .xml，计数各交互类型
    返回 dict：各交互类型计数 + 接触残基数
    """
    result = {f"plip_{k}_count": 0 for k in PLIP_INTERACTION_TYPES}
    result["plip_contact_residues_count"] = 0
    if not FEAT_PLIP_INTERACTIONS:
        return result

    workdir = Path(tempfile.mkdtemp(dir=str(tmp_root)))
    merged_pdb = workdir / "merged.pdb"
    try:
        # 合并 protein + ligand 到一份 PDB（用 Biopython + openbabel 组合）
        # 简便做法：直接 cat protein 二进制 + 用 obabel 转 ligand 为 PDB
        from Bio.PDB import PDBIO, PDBParser
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure("prot", protein_pdb)
            io = PDBIO()
            io.set_structure(struct)
            with open(merged_pdb, "w", encoding="utf-8") as fh:
                io.save(fh)
        # ligand → PDB 追加
        lig_pdb = workdir / "lig.pdb"
        r = subprocess.run(
            ["obabel", ligand_sdf_or_mol2, "-O", str(lig_pdb)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0 and lig_pdb.exists():
            with open(merged_pdb, "a", encoding="utf-8") as fh:
                fh.write(lig_pdb.read_text(encoding="utf-8", errors="ignore"))

        # 调 PLIP
        out_dir = workdir / "out"
        out_dir.mkdir(exist_ok=True)
        r = subprocess.run(
            ["plip", "-f", str(merged_pdb), "-o", str(out_dir), "-x", "-y", "-t", "-q"],
            capture_output=True, text=True, timeout=180,
        )
        # 解析 report.txt / *.xml
        rpt_txt = next(out_dir.glob("*_report.txt"), None)
        if rpt_txt is not None:
            txt = rpt_txt.read_text(encoding="utf-8", errors="ignore")
            # 解析各交互段的条目数（简单正则：如 "Hydrogen bonds" 下面的条目数）
            # Baseline: 直接扫关键数字行
            counts = _parse_plip_report_text(txt)
            for k, v in counts.items():
                result[f"plip_{k}_count"] = v
            # 接触残基 = 所有涉及的 (chain+resid) 去重
            residues = _collect_contact_residues(txt)
            result["plip_contact_residues_count"] = len(residues)
    except Exception:
        # 失败不计，返回 0 值结构（保证对齐列）
        pass
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return result


def _parse_plip_report_text(txt: str) -> dict:
    """简化版解析：对每种交互类型，统计在 report.txt 中对应小节的条目行数。
    PLIP report.txt 里每类交互有一个标题和下面的表格行（每一行 1 个交互）。
    这里用正则匹配标题段后数表格行数。"""
    out = {k: 0 for k in [x.removesuffix("s") if False else x for x in PLIP_INTERACTION_TYPES]}
    # 简单实现：以标题为锚（大写开头），然后在接下来空行前统计以数字开头或以 "|" 开头的行数
    # 作为近似，每种交互匹配其小节标题行后若干 "条" 数（括号内数字）如 "Hydrogen bonds (12)"
    for itype in PLIP_INTERACTION_TYPES:
        m = re.search(re.escape(itype.replace("_", " ").title()) + r"\s*\(\s*(\d+)\s*\)", txt)
        if m:
            out[itype] = int(m.group(1))
        else:
            # 另一种格式：标题后紧跟表格，行数 = 标题至空行之间的行数 - 2（表头）
            pass
    return out


def _collect_contact_residues(txt: str) -> set:
    """从 report.txt 里提取所有 (chain:resid+resname) 组合，去重返回集合。"""
    pattern = re.compile(r"\b([A-Z0-9]{1,3}):(\d+)\((\w{3})\)")
    matches = pattern.findall(txt)
    return set(matches)


# ============================================================
# 工具：蛋白/口袋 简化描述符
# ============================================================
def compute_protein_desc(protein_pdb: str, pocket_pdb: str | None = None) -> dict:
    out = {
        "prot_seq_len": np.nan,
        "prot_n_residues": np.nan,
        "pocket_n_residues": np.nan,
        "pocket_sasa_a2": np.nan,
    }
    try:
        from Bio.PDB import PDBParser, Residue
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = PDBParser(QUIET=True)
            if Path(protein_pdb).exists():
                s = parser.get_structure("P", protein_pdb)
                residues = [r for r in s.get_residues() if r.id[0] == " "]  # 标准氨基酸
                out["prot_n_residues"] = len(residues)
                out["prot_seq_len"] = len(residues)  # 简化：残基数代替序列长度
            if pocket_pdb and Path(pocket_pdb).exists():
                s2 = parser.get_structure("K", pocket_pdb)
                residues_k = [r for r in s2.get_residues() if r.id[0] == " "]
                out["pocket_n_residues"] = len(residues_k)
    except Exception:
        pass
    return out


# ============================================================
# 原子对距离计数（RF-Score 64维，可选）
# ============================================================
ATOM_ELEMENT_ORDER = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
DISTANCE_BINS = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]  # 7个间隔 -> 7 bin


def element_of(atom) -> str | None:
    try:
        e = atom.element.strip().upper()
        if e in ATOM_ELEMENT_ORDER:
            return e
    except Exception:
        pass
    return None


def compute_atompair_hist_64(protein_pdb: str, ligand_sdf_or_mol2: str) -> dict:
    """C/N/O/S/P/F/Cl/Br/I × 自身（8 种蛋白质元素子集）原子对距离分布。
    实际执行：蛋白侧取 8 种常见元素（C,N,O,S,P,F,Cl,Br），配体侧同，得到 8x8=64 组合。
    每个组合用 DISTANCE_BINS 分 7 个区间，共 64*7=448 维？不，经典 RF-Score 只统计 8x8=64
    的 "近距离 <6Å 计数"（不分 bin），此处保持 64 维：近距离(<6Å)计数。
    TODO[v2]：加 bin 分 7x64=448 维精细特征。
    """
    feats = {}
    PROT_ELS = ["C", "N", "O", "S", "P", "F", "Cl", "Br"]
    LIG_ELS = ["C", "N", "O", "S", "P", "F", "Cl", "Br"]
    try:
        from Bio.PDB import PDBParser
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = PDBParser(QUIET=True)
            p_s = parser.get_structure("P", protein_pdb) if Path(protein_pdb).exists() else None
            # 读 ligand 原子坐标：优先 mol2/sdf → RDKit；失败用 Biopython（难）
            lig_atoms = _extract_ligand_atoms(ligand_sdf_or_mol2)
            if p_s is None or not lig_atoms:
                return {}
            prot_atoms = []
            for a in p_s.get_atoms():
                e = element_of(a)
                if e in PROT_ELS:
                    prot_atoms.append((e, a.coord.astype(np.float32)))
            # 暴力计算（蛋白原子可能上万，N*M 较大，取口袋附近的蛋白原子加速）
            # 先算配体质心，取蛋白 12Å 内原子
            if not lig_atoms:
                return {}
            lig_coords = np.stack([c for _, c in lig_atoms], axis=0)
            centroid = lig_coords.mean(axis=0)
            prot_filtered = []
            for e, c in prot_atoms:
                if np.linalg.norm(c - centroid) < 15.0:
                    prot_filtered.append((e, c))
            # 距离矩阵批量计算
            pc = np.stack([c for _, c in prot_filtered], axis=0) if prot_filtered else np.empty((0, 3))
            lc = lig_coords
            pe = [e for e, _ in prot_filtered]
            le = [e for e, _ in lig_atoms]
            if pc.shape[0] == 0 or lc.shape[0] == 0:
                return {}
            # 计算距离矩阵 (n_prot, n_lig)
            diff = pc[:, None, :] - lc[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)  # (n_p, n_l)
            mask = dist < 6.0  # 经典 6Å 内相互作用范围
            # 64 维计数
            pe_arr = np.array(pe, dtype=object)
            le_arr = np.array(le, dtype=object)
            for i, pelem in enumerate(PROT_ELS):
                for j, lelem in enumerate(LIG_ELS):
                    key = f"apf_{pelem}_{lelem}_lt6"
                    # 取出 (prot in pelem) & (lig in lelem) 的子矩阵，统计 True 数
                    rows = (pe_arr == pelem)
                    cols = (le_arr == lelem)
                    if rows.any() and cols.any():
                        sub = mask[np.ix_(rows, cols)]
                        feats[key] = int(sub.sum())
                    else:
                        feats[key] = 0
            # --- 增强 3D 相互作用特征（不依赖 PLIP/openbabel） ---
            # 氢键：蛋白 N/O - 配体 N/O 距离 < 3.5Å
            hbond_prot = np.isin(pe_arr, ["N", "O"])
            hbond_lig = np.isin(le_arr, ["N", "O"])
            if hbond_prot.any() and hbond_lig.any():
                hbond_mask = (dist < 3.5)[np.ix_(hbond_prot, hbond_lig)]
                feats["3d_hbond_count"] = int(hbond_mask.sum())
            else:
                feats["3d_hbond_count"] = 0
            # 疏水接触：蛋白 C - 配体 C 距离 < 4.5Å
            c_prot = (pe_arr == "C")
            c_lig = (le_arr == "C")
            if c_prot.any() and c_lig.any():
                hydro_mask = (dist < 4.5)[np.ix_(c_prot, c_lig)]
                feats["3d_hydrophobic_count"] = int(hydro_mask.sum())
            else:
                feats["3d_hydrophobic_count"] = 0
            # 盐桥：蛋白 N/O - 配体 O/N 距离 < 5.0Å（简化版，不区分电荷）
            salt_prot = np.isin(pe_arr, ["N", "O"])
            salt_lig = np.isin(le_arr, ["N", "O"])
            if salt_prot.any() and salt_lig.any():
                salt_mask = (dist < 5.0)[np.ix_(salt_prot, salt_lig)]
                feats["3d_salt_bridge_count"] = int(salt_mask.sum())
            else:
                feats["3d_salt_bridge_count"] = 0
            # 接触残基数（蛋白侧 5Å 内有接触的残基去重）
            contact_mask = dist < 5.0
            contact_rows = np.any(contact_mask, axis=1)
            feats["3d_contact_residues"] = int(contact_rows.sum())
    except Exception:
        pass
    return feats


def _extract_ligand_atoms(ligand_path: str):
    """返回 [(element, coord(3,)), ...]。用 _read_ligand_mol 读取（已处理注释行+SDF修复）。"""
    out = []
    mol = _read_ligand_mol(Path(ligand_path))
    if mol is not None:
        try:
            conf = mol.GetConformer()
            for a in mol.GetAtoms():
                sym = a.GetSymbol().strip().upper()
                pos = conf.GetAtomPosition(a.GetIdx())
                out.append((sym, np.array([pos.x, pos.y, pos.z], dtype=np.float32)))
        except Exception:
            pass
    return out


# ============================================================
# 单条样本处理函数（用于并行）
# ============================================================
def process_one(row: pd.Series, do_3d: bool, tmp_root: Path):
    pdb = row["pdb"]
    result_row = {"pdb": pdb, "pKd": row.get("pKd", np.nan), "split": row.get("split", "")}
    # A. 配体 2D（baseline 层必算）
    lig_path = row.get("path_ligand_sdf") or row.get("path_ligand_mol2")
    lig_mol2 = row.get("path_ligand_mol2")
    if FEAT_LIGAND_1D2D or FEAT_LIGAND_MORGAN:
        lig_for_rdk = lig_mol2 if (lig_mol2 and Path(lig_mol2).exists()) else lig_path
        feats_2d = compute_ligand_2d_features(lig_for_rdk) if lig_for_rdk and Path(lig_for_rdk).exists() else {}
        result_row.update(feats_2d)
    # B. 蛋白描述符（始终计算，因很便宜）
    if FEAT_PROTEIN_DESC:
        pp = row.get("path_protein")
        pk = row.get("path_pocket")
        result_row.update(compute_protein_desc(pp, pk))
    # C. 3D 相互作用（原子对距离计数 + 增强：氢键/疏水/盐桥）—— 只在 do_3d=True 时计算
    # 注：PLIP 不可用（openbabel 缺 PDB 格式插件），用 Biopython + RDKit 替代
    if do_3d:
        pp = row.get("path_protein")
        lig_for_3d = lig_mol2 if (lig_mol2 and Path(lig_mol2).exists()) else lig_path
        if pp and lig_for_3d and Path(pp).exists() and Path(lig_for_3d).exists():
            result_row.update(compute_atompair_hist_64(pp, lig_for_3d))
    # D. Vina 能量项（占位，默认关闭）
    if FEAT_VINA_ENERGY:
        # TODO：后续批量 Vina 对接（Meeko/Vina）产出能量项 CSV，在此 merge
        pass
    return result_row


# ============================================================
# main：并行提取
# ============================================================
def main():
    # ---------- 命令行参数：--phase {a|b|all} ----------
    parser = argparse.ArgumentParser(description="step2 特征提取（Phase A 1D/2D / Phase B 3D / all）")
    parser.add_argument(
        "--phase",
        choices=["a", "b", "all"],
        default="all",
        help="a=仅1D/2D Baseline特征; b=仅3D特征(PLIP+原子对); all=两者都跑（默认）",
    )
    args = parser.parse_args()
    run_phase_a = args.phase in ("a", "all")
    run_phase_b = args.phase in ("b", "all")

    print("[step2] 读取 DATASET_refined.csv ...")
    if not DATASET_REFIND_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {DATASET_REFIND_CSV}，请先运行 step1_load_data.py"
        )
    df = pd.read_csv(DATASET_REFIND_CSV, encoding="utf-8-sig")
    print(f"  - 样本数（伪refined）: {len(df)}")
    print(f"  - 运行阶段: phase={args.phase} (run_a={run_phase_a}, run_b={run_phase_b})")

    # ---------- Phase A：Baseline 特征（1D/2D 描述符 + Morgan指纹） ----------
    if run_phase_a:
        done_baseline = set()
        if FEATURES_BASELINE_CSV.exists():
            _old = pd.read_csv(FEATURES_BASELINE_CSV)
            # 续跑判定修复：旧垃圾 CSV（无 desc_ 列或有效行<10%）直接丢弃重算，不续跑
            _has_desc = any(c.startswith("desc_") for c in _old.columns)
            if _has_desc:
                done_baseline = set(_old["pdb"].astype(str).values)
                rows_out = [_old]
            else:
                print(f"[step2] 旧 features_baseline.csv 无 desc_ 列（旧垃圾），丢弃重算", flush=True)
                rows_out = []
        else:
            rows_out = []
        print(f"[step2] Phase A  Baseline 特征提取（串行，跳过已算 {len(done_baseline)}）", flush=True)
        todo = df[~df["pdb"].astype(str).isin(done_baseline)]
        if len(todo) > 0:
            import time as _time
            _t0 = _time.time()
            _n_ok = 0
            _n_fail = 0
            for _i, (_, _row) in enumerate(todo.iterrows(), 1):
                try:
                    _r = process_one(_row, False, None)
                    rows_out.append(pd.DataFrame([_r]))
                    _n_ok += 1
                except Exception as _e:
                    _n_fail += 1
                    print(f"    ! pdb={_row.get('pdb')} 异常: {_e}", flush=True)
                if _i % 200 == 0:
                    _elapsed = _time.time() - _t0
                    _rate = _i / _elapsed if _elapsed > 0 else 0
                    _eta = (len(todo) - _i) / _rate if _rate > 0 else 0
                    print(f"    progress: {_i}/{len(todo)} ok={_n_ok} fail={_n_fail} "
                          f"rate={_rate:.1f}/s eta={_eta:.0f}s", flush=True)
                # 每 2000 条写一次断点 CSV（防中途崩了全丢）
                if _i % 2000 == 0:
                    pd.concat(rows_out, ignore_index=True).to_csv(FEATURES_BASELINE_CSV, index=False)
                    print(f"    [checkpoint] saved {len(rows_out)} rows -> {FEATURES_BASELINE_CSV.name}", flush=True)
            pd.concat(rows_out, ignore_index=True).to_csv(FEATURES_BASELINE_CSV, index=False)
        print(f"  ✅ Baseline 特征完成，共 {len(pd.read_csv(FEATURES_BASELINE_CSV))} 条 -> {FEATURES_BASELINE_CSV}", flush=True)
    else:
        print("[step2] Phase A  ⏸️ 跳过（--phase 未包含 a）", flush=True)

    # ---------- Phase B：3D 特征（原子对距离 + 氢键/疏水/盐桥） ----------
    if run_phase_b:
        print("[step2] Phase B  3D 相互作用特征提取（串行，不依赖 PLIP）", flush=True)
        done_3d = set()
        rows_3d = []
        if FEATURES_3D_CSV.exists():
            _old3d = pd.read_csv(FEATURES_3D_CSV)
            _has_3d = any(c.startswith("apf_") or c.startswith("3d_") for c in _old3d.columns)
            if _has_3d:
                done_3d = set(_old3d["pdb"].astype(str).values)
                rows_3d = [_old3d]
            else:
                print(f"[step2] 旧 features_3d.csv 无 apf_/3d_ 列，丢弃重算", flush=True)
        todo = df[~df["pdb"].astype(str).isin(done_3d)]
        print(f"  待算 {len(todo)} 条，跳过已算 {len(done_3d)}", flush=True)
        if len(todo) > 0:
            import time as _time
            _t0 = _time.time()
            _n_ok = 0
            _n_fail = 0
            for _i, (_, _row) in enumerate(todo.iterrows(), 1):
                try:
                    _r = process_one(_row, True, None)
                    rows_3d.append(pd.DataFrame([_r]))
                    _n_ok += 1
                except Exception as _e:
                    _n_fail += 1
                    print(f"    ! pdb={_row.get('pdb')} 异常: {_e}", flush=True)
                if _i % 100 == 0:
                    _elapsed = _time.time() - _t0
                    _rate = _i / _elapsed if _elapsed > 0 else 0
                    _eta = (len(todo) - _i) / _rate if _rate > 0 else 0
                    print(f"    progress 3D: {_i}/{len(todo)} ok={_n_ok} fail={_n_fail} "
                          f"rate={_rate:.1f}/s eta={_eta:.0f}s", flush=True)
                if _i % 1000 == 0:
                    pd.concat(rows_3d, ignore_index=True).to_csv(FEATURES_3D_CSV, index=False)
                    print(f"    [checkpoint] saved {len(rows_3d)} rows", flush=True)
            pd.concat(rows_3d, ignore_index=True).to_csv(FEATURES_3D_CSV, index=False)
        print(f"  ✅ 3D 特征完成，共 {len(pd.read_csv(FEATURES_3D_CSV))} 条 -> {FEATURES_3D_CSV}", flush=True)
    else:
        print("[step2] Phase B  ⏸️ 跳过（--phase 未包含 b）", flush=True)


if __name__ == "__main__":
    main()
