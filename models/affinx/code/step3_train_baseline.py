# step3_train_baseline.py —— Baseline 模型训练 + 5 折 CV 评估
# 读取：features_baseline.csv（step2产出）
# 产出：
#   models/xgboost_baseline.pkl        —— Baseline 模型（在全训练集上重训练最后版本）
#   reports/baseline_cv_report.txt     —— 5折CV 指标（R²/RMSE/MAE）
#   reports/baseline_split_metrics.csv —— 每折明细
#   reports/feature_importance_baseline.png —— Top30 重要特征柱状图（可视化素材）
#
# 说明：
# - Baseline 只用 1D/2D 描述符 + Morgan 指纹（不用 3D），用来量化"加3D特征能提升多少"
# - 超参数用 config.XGB_BASE_PARAMS（默认参数，不调参），后续 step4 用 Optuna 调
# - 偏置 baseline 对比实验：单独训练一个"只用配体特征，不加蛋白描述符"的对照模型，
#   用来判断模型是否真学到了蛋白-配体相互作用，还是只是学到数据集偏置。

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from config import (
    FEATURES_DIR, MODELS_DIR, REPORTS_DIR,
    XGB_BASE_PARAMS, N_SPLITS_CV, RANDOM_SEED,
    METRIC_THRESHOLDS,
)

FEATURES_BASELINE_CSV = FEATURES_DIR / "features_baseline.csv"
FEATURES_3D_CSV = FEATURES_DIR / "features_3d.csv"
MODEL_BASENAME = "xgboost_baseline_3d" if FEATURES_3D_CSV.exists() else "xgboost_baseline"


def _load_features(csv_path: Path):
    df = pd.read_csv(csv_path)
    # 去掉 pKd NaN、split 空的行
    df = df.dropna(subset=["pKd", "split"])
    # 特征列：除 pdb / pKd / split 之外的数值列
    meta_cols = {"pdb", "pKd", "split"}
    feature_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    # 填充 NaN + inf（RDKit MaxPartialCharge/MinPartialCharge 对某些分子返回 inf）
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, feature_cols


def _make_X_y_split(df, feature_cols):
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    def Xy(d):
        return d[feature_cols].values, d["pKd"].values
    return Xy(train), Xy(val), Xy(test), train, val, test


def _train_xgboost(X, y):
    from xgboost import XGBRegressor
    m = XGBRegressor(**XGB_BASE_PARAMS)
    m.fit(X, y)
    return m


def _eval(y_true, y_pred):
    return {
        "r2":   r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  mean_absolute_error(y_true, y_pred),
    }


def _cv_on_trainval(train_val_df, feature_cols) -> pd.DataFrame:
    """在 train+val 混合的 5 折 CV 上报告每折性能（传统ML报告方式，与文献对齐）。
    同时保留两种口径：(1) 固定 split 评估；(2) 5折CV 评估，结果都报告。
    """
    from xgboost import XGBRegressor
    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
    X = train_val_df[feature_cols].values
    y = train_val_df["pKd"].values
    folds = []
    for i, (tr, va) in enumerate(kf.split(X), 1):
        m = XGBRegressor(**XGB_BASE_PARAMS)
        m.fit(X[tr], y[tr])
        pr = m.predict(X[va])
        metrics = _eval(y[va], pr)
        metrics["fold"] = i
        metrics["n_train"] = len(tr)
        metrics["n_val"] = len(va)
        folds.append(metrics)
    return pd.DataFrame(folds)


def _plot_feature_importance(model, feature_names, top_n=30, out_png: Path | None = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(order)), importances[order], color="#2b7bbd")
        plt.yticks(range(len(order)), [feature_names[i] for i in order], fontsize=8)
        plt.xlabel("F score (XGBoost feature importance)")
        plt.title(f"Top-{top_n} Feature Importance (Baseline XGBoost)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if out_png is not None:
            plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception:
        pass


def main():
    print(f"[step3] 加载特征 (model={MODEL_BASENAME}) ...")
    if not FEATURES_BASELINE_CSV.exists():
        raise FileNotFoundError(f"找不到 {FEATURES_BASELINE_CSV}，请先运行 step2_extract_features.py")
    df, feature_cols = _load_features(FEATURES_BASELINE_CSV)
    print(f"  - baseline 样本数: {len(df)}, 特征数: {len(feature_cols)}")
    # 如果 3D 特征也存在，按 pdb merge
    if FEATURES_3D_CSV.exists():
        print("  - 检测到 3D 特征，合并中 ...")
        df_3d, feat_3d_cols = _load_features(FEATURES_3D_CSV)
        cols_3d = [c for c in feat_3d_cols if c not in df.columns]
        df = df.merge(df_3d[["pdb"] + cols_3d], on="pdb", how="left")
        df[cols_3d] = df[cols_3d].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feature_cols = feature_cols + cols_3d
        print(f"  - 合并 3D 特征 {len(cols_3d)} 列，总特征数: {len(feature_cols)}")
    print(f"  - 最终样本数: {len(df)}, 特征数: {len(feature_cols)}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), train_df, val_df, test_df = _make_X_y_split(df, feature_cols)

    # -------- 5 折 CV（train+val 混合，做文献对齐的报告） --------
    print("[step3] 5折交叉验证（train+val）...")
    trainval = pd.concat([train_df, val_df], ignore_index=True)
    cv_df = _cv_on_trainval(trainval, feature_cols)
    cv_df.to_csv(REPORTS_DIR / f"{MODEL_BASENAME}_cv_folds.csv", index=False)
    r2_mean = cv_df["r2"].mean()
    rmse_mean = cv_df["rmse"].mean()
    mae_mean = cv_df["mae"].mean()
    print(f"  CV 5-fold 平均: R²={r2_mean:.3f}, RMSE={rmse_mean:.3f}, MAE={mae_mean:.3f}")

    # -------- 固定 split 下训练 / 评估 --------
    print("[step3] 固定 split 训练 + 独立评估 ...")
    model = _train_xgboost(X_train, y_train)
    pred_tr = model.predict(X_train)
    pred_vl = model.predict(X_val)
    pred_ts = model.predict(X_test)
    metrics_fixed = pd.DataFrame([
        {"split": "train", **_eval(y_train, pred_tr)},
        {"split": "val",   **_eval(y_val,   pred_vl)},
        {"split": "test",  **_eval(y_test,  pred_ts)},
    ])
    metrics_fixed.to_csv(REPORTS_DIR / f"{MODEL_BASENAME}_split_metrics.csv", index=False)
    print(metrics_fixed.to_string(index=False))

    # -------- 偏置 baseline：检测模型是否只是学到数据集偏置（Durant 2025 警示应对） --------
    # 偏置策略分两种场景：
    #   场景 A（1D/2D baseline，当前 step3）：无 prot_* / pocket_* 蛋白特征列
    #       → 偏置 baseline 只用 Morgan 指纹 fp_*，对比 desc_ 描述符是否提供额外信息
    #         若 ΔR² < 0.05 → 描述符基本没用，模型主要靠子结构偏置
    #   场景 B（3D 特征，step4）：有 prot_* / pocket_* 蛋白相关列
    #       → 偏置 baseline 去掉蛋白列，仅保留配体自身描述符/指纹
    has_protein_cols = any(c.startswith("prot_") or c.startswith("pocket_") for c in feature_cols)
    if has_protein_cols:
        # 场景 B：去掉蛋白相关列（保留配体列）
        bias_cols = [c for c in feature_cols if not c.startswith("prot_") and not c.startswith("pocket_")]
        bias_label = "ligand_only_rm_protein"
        print(f"[step3] 偏置对照实验(场景B)：去除蛋白/pocket列，保留 {len(bias_cols)} 配体特征")
    else:
        # 场景 A：baseline 无蛋白列，只用 Morgan 指纹(fp_*) 做偏置基线
        bias_cols = [c for c in feature_cols if c.startswith("fp_")]
        bias_label = "morgan_fp_only"
        print(f"[step3] 偏置对照实验(场景A)：仅用 Morgan 指纹 ({len(bias_cols)} 列) 作为偏置基线")
    if len(bias_cols) < 10:
        print(f"  ⚠️ 偏置基线特征数过少 ({len(bias_cols)})，跳过偏置对照")
    else:
        try:
            from xgboost import XGBRegressor
            X_tv_bias = trainval[bias_cols].values
            y_tv = trainval["pKd"].values
            kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
            r2s = []
            for tr, va in kf.split(X_tv_bias):
                mb = XGBRegressor(**XGB_BASE_PARAMS)
                mb.fit(X_tv_bias[tr], y_tv[tr])
                r2s.append(r2_score(y_tv[va], mb.predict(X_tv_bias[va])))
            r2_mean_bias = float(np.mean(r2s))
            print(f"  偏置模型({bias_label}) CV R²: {r2_mean_bias:.3f}")
            print(f"  完整模型 vs 偏置模型 ΔR²: {r2_mean - r2_mean_bias:+.3f}")
            pd.DataFrame([{
                "model": "full_baseline",
                "cv_r2_mean": r2_mean,
            }, {
                "model": f"bias_{bias_label}",
                "cv_r2_mean": r2_mean_bias,
            }, {
                "model": "delta",
                "cv_r2_mean": r2_mean - r2_mean_bias,
            }]).to_csv(REPORTS_DIR / f"{MODEL_BASENAME}_bias_comparison.csv", index=False)
        except Exception as e:
            print(f"  ! 偏置对照实验失败: {e}")

    # -------- 模型保存（全 train+val 重训练最后一版） --------
    print("[step3] 在全 train+val 上重训练并保存 ...")
    X_tv = trainval[feature_cols].values
    y_tv = trainval["pKd"].values
    final = _train_xgboost(X_tv, y_tv)
    with open(MODELS_DIR / f"{MODEL_BASENAME}.pkl", "wb") as fh:
        pickle.dump({"model": final, "feature_cols": feature_cols, "params": XGB_BASE_PARAMS}, fh)
    print(f"  模型已保存: {MODELS_DIR / f'{MODEL_BASENAME}.pkl'}")

    # -------- 特征重要性图（无显示环境可能 crash，用 try/except） --------
    try:
        _plot_feature_importance(
            final, feature_cols, top_n=30,
            out_png=REPORTS_DIR / f"{MODEL_BASENAME}_feature_importance_top30.png",
        )
    except Exception as _e:
        print(f"  [WARN] 特征重要性图绘制失败（不影响核心结果）: {_e}")

    # -------- 汇总报告 --------
    report_lines = [
        f"# Baseline XGBoost 训练报告",
        f"训练样本（train+val）: {len(trainval)}",
        f"独立测试样本: {len(test_df)}",
        f"特征数: {len(feature_cols)}",
        "",
        f"## 5折CV平均",
        f"- R² = {r2_mean:.3f}",
        f"- RMSE = {rmse_mean:.3f} (log 单位)",
        f"- MAE = {mae_mean:.3f} (log 单位)",
        "",
        f"## 合格标准（阈值来自 config.METRIC_THRESHOLDS）",
        f"- CV R² > 0.6 (config 预设阈值): {'✅' if r2_mean > METRIC_THRESHOLDS['cv_r2_good'] else '⚠️' if r2_mean > METRIC_THRESHOLDS['cv_r2_min'] else '❌'}",
        f"- CV RMSE < 1.4 (log): {'✅' if rmse_mean < METRIC_THRESHOLDS['cv_rmse_good'] else '⚠️'}",
        "",
        f"## 固定 split 独立评估",
        metrics_fixed.to_string(index=False),
    ]
    with open(REPORTS_DIR / f"{MODEL_BASENAME}_cv_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))
    print(f"  报告: {REPORTS_DIR / f'{MODEL_BASENAME}_cv_report.txt'}")
    print("  ✅ step3 完成。下一步：若 baseline 达标 → step4 加3D特征+Optuna调参；不达标 → 先查特征/数据质量。")


if __name__ == "__main__":
    main()
