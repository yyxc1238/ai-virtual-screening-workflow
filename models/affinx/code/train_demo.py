# -*- coding: utf-8 -*-
"""
train_demo.py —— 小规模"训练 → 超参优化 → 验证 → 推理"可运行教学示例
=====================================================================
正式训练脚本 step4_optimized_v3.py 面向 13,770 条 PDBBind 训练池 +
Optuna 50-trial 搜索 + 5 折 CV（需先构建数百 MB 的 3D/Vina 特征表），
普通电脑不适合现场复现。本脚本在**同一套 2,110 特征列、同一种
Δ-learning 建模思路**下，用一份小规模真实抽样特征演示完整闭环：

  1. 数据：--features 指定的 CSV（列含 pdb,pKd,split + 特征列，
     split 取值 train/val）；全量特征表获取方式见上级目录 README.md 的
     数据来源说明。
  2. 训练/优化：3 组超参（max_depth/learning_rate/subsample…）×
     5 折交叉验证，按平均 RMSE 选最优——这是正式管线 Optuna 50-trial
     搜索的确定性缩小版（相同的"CV 选参"逻辑，结果可复现、无需 Optuna）。
  3. 验证：最优超参在全量 train 上重训，对未见过的 val 计算
     RMSE/MAE/R²，并与发布模型记录的 CV_RMSE=1.2318（13,770 条口径）对照。
  4. 推理：用 demo 模型对 --candidates 指定的候选表（mol_id+SMILES）打分，
     并附训练适用域判定（与 predict.py/screen.py 同库）。

注意：demo 模型只学少量样本，泛化显著弱于发布的 13,770 条 v3 模型，
      其输出仅用于教学演示；正式候选打分请用 predict.py / screen.py。

运行（约 10-30 秒，纯 CPU）：
  python train_demo.py --features features_demo.csv --candidates candidates.csv
输出：
  demo_metrics.json     —— CV/验证指标与最优超参
  demo_predictions.csv  —— 候选的 demo 模型打分
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict import compute_2d_features
from applicability_domain import FpReference, assess as ad_assess

TARGET = "pKd"
N_SPLITS = 5
SEED = 42
N_ESTIMATORS = 400

# 确定性小网格：正式 step4 用 Optuna 在更大空间里搜 50 个 trial，
# 这里压缩为 3 个有代表性的配置（浅树高学习率 ↔ 深树低学习率）。
PARAM_GRID = [
    {"max_depth": 3, "learning_rate": 0.10, "subsample": 0.8,
     "colsample_bytree": 0.8, "min_child_weight": 5},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8,
     "colsample_bytree": 0.8, "min_child_weight": 5},
    {"max_depth": 8, "learning_rate": 0.03, "subsample": 0.8,
     "colsample_bytree": 0.8, "min_child_weight": 3},
]


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main():
    ap = argparse.ArgumentParser(description="小规模训练-评估-推理教学闭环")
    ap.add_argument("--features", required=True,
                    help="特征抽样 CSV（列：pdb,pKd,split + 特征列）")
    ap.add_argument("--candidates", default=None,
                    help="可选：候选表 CSV（mol_id, smiles[, vina_docking_score]），"
                         "提供则做 demo 推理")
    ap.add_argument("--out-dir", default=".", help="输出目录")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from xgboost import XGBRegressor
    from sklearn.model_selection import KFold

    print("=" * 70)
    print("步骤 1/4  读取特征抽样（train + val）")
    print("=" * 70)
    data_path = Path(args.features)
    df = pd.read_csv(data_path)
    feat_cols = [c for c in df.columns if c not in ("pdb", TARGET, "split")]
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "val"].reset_index(drop=True)
    Xtr, ytr = tr[feat_cols].astype(np.float32), tr[TARGET].astype(np.float32)
    Xva, yva = va[feat_cols].astype(np.float32), va[TARGET].astype(np.float32)
    print(f"  特征维度 {len(feat_cols)}（与发布 v3 模型同口径）")
    print(f"  train {len(tr)} | val {len(va)} | pKd 范围 "
          f"{df[TARGET].min():.2f}-{df[TARGET].max():.2f}")
    print("  说明：3D/Vina 缺失值保留 NaN，由树模型原生分裂处理（与 step4 一致）。")

    print("\n" + "=" * 70)
    print("步骤 2/4  5 折 CV 超参优化（Optuna 搜索的确定性缩小版）")
    print("=" * 70)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_results = []
    for k, params in enumerate(PARAM_GRID, 1):
        fold_rmse = []
        for fi, (itr, ivl) in enumerate(kf.split(Xtr), 1):
            m = XGBRegressor(n_estimators=N_ESTIMATORS,
                             objective="reg:squarederror",
                             tree_method="hist", random_state=SEED,
                             n_jobs=2, **params)
            m.fit(Xtr.iloc[itr], ytr.iloc[itr], verbose=False)
            fold_rmse.append(rmse(ytr.iloc[ivl].to_numpy(),
                                  m.predict(Xtr.iloc[ivl])))
        cv_results.append({"params": params,
                           "cv_rmse_mean": float(np.mean(fold_rmse)),
                           "cv_rmse_std": float(np.std(fold_rmse)),
                           "fold_rmse": [round(x, 4) for x in fold_rmse]})
        print(f"  配置{k}: depth={params['max_depth']} lr={params['learning_rate']}"
              f" -> CV RMSE {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")
    best = min(cv_results, key=lambda d: d["cv_rmse_mean"])
    print(f"  >> 最优：depth={best['params']['max_depth']} "
          f"lr={best['params']['learning_rate']}（CV RMSE {best['cv_rmse_mean']:.4f}）")

    print("\n" + "=" * 70)
    print("步骤 3/4  全量重训 + 留出 val 评估")
    print("=" * 70)
    final = XGBRegressor(n_estimators=N_ESTIMATORS, objective="reg:squarederror",
                         tree_method="hist", random_state=SEED, n_jobs=2,
                         **best["params"])
    final.fit(Xtr, ytr, verbose=False)
    pred_va = final.predict(Xva)
    val_metrics = {
        "val_rmse": rmse(yva.to_numpy(), pred_va),
        "val_mae": float(np.mean(np.abs(yva.to_numpy() - pred_va))),
        "val_r2": float(1 - np.sum((yva.to_numpy() - pred_va) ** 2)
                   / np.sum((yva.to_numpy() - yva.mean()) ** 2)),
    }
    print(f"  val RMSE {val_metrics['val_rmse']:.4f} | "
          f"MAE {val_metrics['val_mae']:.4f} | R² {val_metrics['val_r2']:.3f}")
    print("  对照：发布 v3（13,770 条训练池）5 折 CV RMSE = 1.2318；"
          "demo 样本量小，误差更大、波动更明显，符合预期。")

    rows = []
    if args.candidates:
        print("\n" + "=" * 70)
        print("步骤 4/4  demo 模型对候选表推理（+适用域）")
        print("=" * 70)
        try:
            fp_ref = FpReference()
        except FileNotFoundError:
            fp_ref = None
        cand = pd.read_csv(args.candidates, dtype={"mol_id": str})
        for _, r in cand.iterrows():
            feats = compute_2d_features(str(r["smiles"]).strip())
            if feats is None:
                continue
            feats["prot_seq_len"] = float(r["prot_seq_len"]) if "prot_seq_len" in cand.columns else 0.0
            feats["prot_n_residues"] = float(r["prot_n_residues"]) if "prot_n_residues" in cand.columns else 0.0
            feats["pocket_n_residues"] = float(r["pocket_n_residues"]) if "pocket_n_residues" in cand.columns else 0.0
            feats["vina_total"] = (float(r["vina_docking_score"])
                                   if pd.notna(r.get("vina_docking_score")) else 0.0)
            x = np.array([[feats.get(c, np.nan) for c in feat_cols]], dtype=np.float32)
            p = float(final.predict(x)[0])
            row = {"mol_id": r["mol_id"], "demo_pKd": round(p, 4),
                   "demo_IC50_uM": round((10 ** (-p)) * 1e6, 2)}
            if fp_ref is not None:
                ad = ad_assess(str(r["smiles"]).strip(), fp_ref)
                if ad:
                    row["domain_flag"] = ad["domain_flag"]
                    row["domain_max_tanimoto"] = ad["domain_max_tanimoto"]
            rows.append(row)
        out_pred = pd.DataFrame(rows)
        print(out_pred.to_string(index=False))
    else:
        out_pred = pd.DataFrame(rows)

    metrics = {
        "data": {"train_rows": len(tr), "val_rows": len(va), "n_features": len(feat_cols),
                 "source": data_path.name,
                 "note": "教学小样本，非发布模型；发布模型训练池 13,770 条"},
        "cv": {"n_splits": N_SPLITS, "n_estimators": N_ESTIMATORS,
               "grid_results": cv_results, "best_params": best["params"]},
        "holdout_val": val_metrics,
        "released_v3_reference": {"cv_rmse": 1.2318, "train_pool": 13770},
    }
    with open(out_dir / "demo_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    out_pred.to_csv(out_dir / "demo_predictions.csv",
                    index=False, encoding="utf-8-sig")
    print(f"\n[train_demo] 指标 -> {out_dir / 'demo_metrics.json'}")
    print(f"[train_demo] 推理 -> {out_dir / 'demo_predictions.csv'}")
    print("[train_demo] 教学闭环完成。正式 13,770 条训练+Optuna 见 "
          "step4_optimized_v3.py。")


if __name__ == "__main__":
    main()
