# -*- coding: utf-8 -*-
"""
train.py —— AffinX（XGBoost v3 结合强度模型）训练编排入口
=====================================================================
按原始训练顺序一键执行：

  step1_load_data.py        解析 PDBBind v2020.R1 INDEX + 结构目录 -> DATASET_refined.csv
  step2_extract_features.py 2D 描述符 + Morgan 指纹 + 3D 相互作用特征
  (Vina 能量项合并)          features_3d_vina_5316.csv —— 需先完成批量 Vina 对接并
                            合并能量项（原管线中的特征合并脚本未随本仓库发布，
                            可按"对接得分列 + vina_available 标记"的表结构自行实现）
  step4_optimized_v3.py     Δ-learning + Optuna 50 trials 调参 + 5 折 CV
                            + CASF-2016 基准 -> xgboost_optimized_v3.pkl

前置条件（数据/外部工具不入包，需自行准备）：
  1. PDBBind v2020.R1（refined/general set，约 30GB 解压结构）。
     数据获取：从 PDBBind 官网（https://www.pdbbind-plus.org.cn 或原学术页）
     下载，遵守其许可，本仓库不做数据再分发。
     设置环境变量 PDBBIND_ROOT 指向解压根目录（其下含 P-L/ 与 index/）。
  2. conda 环境：见 requirements.txt
     （rdkit、openbabel、biopython、plip 3.0.1、xgboost、optuna）。
  3. CASF-2016 core set 285 清单放 <features目录>/CASF2016_core_set_285.dat
     （无此文件时跳过 CASF 基准，不影响模型训练）。

用法：
  # Windows PowerShell
  $env:PDBBIND_ROOT="<PDBBind 解压根目录>"
  python train.py --start 1 --end 4

  # Linux/macOS
  PDBBIND_ROOT=/data/PDBBind_v2020R1 python train.py --start 1 --end 4

产物：code/runs/xgboost_train/{features,models,reports}/（不覆盖随仓库发布的
      ../xgboost_optimized_v3.pkl；请比对 reports 指标后再决定是否替换）。

注意：固定随机种子 RANDOM_SEED=42（config.py），所有数据划分可复现。
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE  # 训练各步骤脚本与本入口同目录
STEPS = {
    1: ("step1_load_data.py", "PDBBind 索引解析与数据集构建"),
    2: ("step2_extract_features.py", "特征提取（2D + Morgan + 3D 相互作用）"),
    4: ("step4_optimized_v3.py", "Δ-learning + Optuna 调参 + CV/CASF 评估"),
}


def check_environment():
    missing = []
    try:
        import rdkit  # noqa
    except ImportError:
        missing.append("rdkit")
    try:
        import xgboost  # noqa
    except ImportError:
        missing.append("xgboost")
    if missing:
        sys.exit(f"[ERROR] 缺少依赖: {', '.join(missing)}，请先按 requirements.txt 建环境。")
    if not os.environ.get("PDBBIND_ROOT"):
        print("=" * 70)
        print("未检测到 PDBBIND_ROOT 环境变量。训练需要 PDBBind v2020.R1 原始数据")
        print("（约 30GB，许可限制不随仓库分发，获取方式见上级目录 README.md）。")
        print("设置后重试：")
        print('  PowerShell : $env:PDBBIND_ROOT="<PDBBind 解压根目录>"')
        print("  bash       : export PDBBIND_ROOT=/data/PDBBind_v2020R1")
        print("=" * 70)
        sys.exit(0)
    root = Path(os.environ["PDBBIND_ROOT"])
    if not (root / "index").exists() and not (root / "P-L").exists():
        sys.exit(f"[ERROR] PDBBIND_ROOT={root} 下未发现 index/ 或 P-L/ 目录，请检查。")


def main():
    ap = argparse.ArgumentParser(description="XGBoost v3 训练编排")
    ap.add_argument("--start", type=int, default=1, choices=[1, 2, 4], help="起始步骤")
    ap.add_argument("--end", type=int, default=4, choices=[1, 2, 4], help="结束步骤")
    args = ap.parse_args()

    check_environment()
    sys.path.insert(0, str(TRAIN_DIR))

    for step_id in sorted(STEPS):
        if step_id < args.start or step_id > args.end:
            continue
        script, desc = STEPS[step_id]
        print("\n" + "#" * 70)
        print(f"# STEP {step_id}: {script} —— {desc}")
        print("#" * 70)
        runpy.run_path(str(TRAIN_DIR / script), run_name="__main__")

    print("\n[train] 全部步骤完成。产物位于 runs/xgboost_train/。")
    print("  与随仓库发布的 ../xgboost_optimized_v3.pkl 比对指标后，")
    print("  再决定是否替换并同步更新 README.md。")


if __name__ == "__main__":
    main()
