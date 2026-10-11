# -*- coding: utf-8 -*-
"""
build_train_fp_ref.py —— 从训练特征表抽取 Morgan 指纹适用域参考库
=====================================================================
输入（不入包，约 116 MB）：
  features_3d_vina_5316.csv（14,055 条 × 2,148 列；由 step2_extract_features.py
  与批量 Vina 对接特征合并生成；PDBBind v2020 数据获取方式见上级目录
  README.md 的数据来源节）

处理（与 step4_optimized_v3.py 完全同口径）：
  排除 CASF-2016 core set 285 个外部基准 PDB（公开基准 ID 清单，
  --casf 传入），剩余 14,055−285 = 13,770 条，即模型实际训练池。

输出（随仓库分发，约 0.8 MB）：
  train_fingerprints_13770.npz（默认写到上级目录，即 ../train_fingerprints_13770.npz）
    pdb           : PDB 复合物 ID
    pkd           : 实验 pKd
    split         : 原特征表内 train/val/test 标记（参考库保留全部 13,770 条，
                    因五折 CV 时它们都曾作为训练折；AD 关心"模型见过的化学空间"）
    bits_packed   : 2048 bit Morgan(r=2) 指纹，np.packbits 压缩 (N, 256) uint8

用途：predict.py / screen.py 的适用域最近邻 Tanimoto 与不确定性区间，
      详见 applicability_domain.py。

用法：
  python build_train_fp_ref.py \
      --features <path-to>/features_3d_vina_5316.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT_PATH = HERE.parents[1] / "train_fingerprints_13770.npz"
FP_COLS = [f"fp_morgan_{i}" for i in range(2048)]
EXPECTED_TRAIN_POOL = 13770


def main():
    ap = argparse.ArgumentParser(description="构建 Morgan 指纹适用域参考库")
    ap.add_argument("--features", required=True,
                    help="features_3d_vina_5316.csv 路径（训练全量特征表）")
    ap.add_argument("--casf", default="",
                    help="CASF-2016 core set 285 ID 清单（tab 分隔，#code 列）；"
                         "缺省时未排除外部基准，仅打印警告")
    ap.add_argument("--output", default=str(OUT_PATH))
    args = ap.parse_args()

    fp = Path(args.features)
    if not fp.exists():
        sys.exit(f"[ERROR] 特征表不存在: {fp}")

    print(f"[fp_ref] 读取 {fp.name} ...")
    df = pd.read_csv(fp, usecols=["pdb", "pKd", "split"] + FP_COLS)
    print(f"[fp_ref] 全量 {len(df)} 条")

    casf = Path(args.casf) if args.casf else None
    if casf is not None and casf.exists():
        dc = pd.read_csv(casf, sep="\t")
        id_col = "#code" if "#code" in dc.columns else "code"
        casf_ids = set(dc[id_col].astype(str).str.lower())
        before = len(df)
        df = df[~df["pdb"].astype(str).str.lower().isin(casf_ids)].reset_index(drop=True)
        print(f"[fp_ref] 排除 CASF-2016 {len(casf_ids)} 个外部基准: {before} -> {len(df)}")
    else:
        print(f"[fp_ref] [WARN] 未提供/未找到 CASF 清单，未排除外部基准")

    if len(df) != EXPECTED_TRAIN_POOL:
        print(f"[fp_ref] [WARN] 训练池行数 {len(df)} != 发布口径 {EXPECTED_TRAIN_POOL}，"
              f"请核对特征表版本（继续写出，但需在 README 说明）")

    bits = df[FP_COLS].to_numpy(dtype=np.uint8)
    uniq = set(np.unique(bits).tolist())
    if not uniq.issubset({0, 1}):
        sys.exit(f"[ERROR] 指纹列出现非 0/1 值: {uniq}")
    packed = np.packbits(bits, axis=1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pdb=df["pdb"].astype(str).to_numpy(),
        pkd=df["pKd"].astype(np.float32).to_numpy(),
        split=df["split"].astype(str).to_numpy(),
        bits_packed=packed,
        morgan_radius=np.int32(2),
        morgan_nbits=np.int32(2048),
        n_train_pool=np.int32(len(df)),
        casf_excluded=np.int32(285),
        source_file=fp.name,
    )
    print(f"[fp_ref] 写出 {out}（{out.stat().st_size/1e6:.2f} MB，解压后 {packed.nbytes/1e6:.2f} MB）")

    # 自检：回读并比对任意 3 行
    z = np.load(out, allow_pickle=True)
    rb = np.unpackbits(z["bits_packed"], axis=1)[:, :2048]
    for i in np.linspace(0, len(df) - 1, 3, dtype=int):
        assert np.array_equal(rb[i], bits[i]), f"回读比对失败 row {i}"
    print("[fp_ref] 回读自检通过（3/3 行一致）。")


if __name__ == "__main__":
    main()
