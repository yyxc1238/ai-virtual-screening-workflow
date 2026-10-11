# -*- coding: utf-8 -*-
"""
Step 4 v3: Delta-learning 优化训练（修正版）

v2失败原因：1612样本对2065特征太少（特征/样本=1.28），泛化能力下降
v3修正策略：
  1. 保留14055全量样本（保留统计能力）
  2. Vina特征用NaN（不填0），XGBoost原生处理缺失值
     -> 无Vina的12000+样本不会被0值噪声污染
     -> 有Vina的1897样本的Vina特征发挥作用
  3. CASF-285严格hold out
  4. max_depth 4-7 + gap惩罚0.3
  5. 50 trials
"""
import sys
import time
import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置（提交版：统一从 config.py 读取，产物落在包内 runs/）
# ============================================================
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURES_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402

FEATURES_VINA_CSV = FEATURES_DIR / "features_3d_vina_5316.csv"
CASF_CSV = FEATURES_DIR / "CASF2016_core_set_285.dat"

MODEL_NAME = "xgboost_optimized_v3"
RANDOM_SEED = 42
N_TRIALS = 50
N_SPLITS = 5
GAP_PENALTY = 0.3

VINA_COLS = ["vina_gauss1", "vina_gauss2", "vina_repulsion",
             "vina_hydrophobic", "vina_hbond", "vina_total"]

print("=" * 70)
print(f"Step 4 v3: Delta-learning 优化训练（修正版）")
print(f"  样本: 14055全量（保留统计能力）")
print(f"  Vina: 用NaN不填0（XGBoost原生处理缺失）")
print(f"  超参: max_depth 4-7 + gap惩罚({GAP_PENALTY})")
print(f"  Optuna: {N_TRIALS} trials")
print("=" * 70)

# ============================================================
# 1. 加载数据
# ============================================================
print(f"\n[1/6] 加载特征文件: {FEATURES_VINA_CSV.name}")
df = pd.read_csv(FEATURES_VINA_CSV)
print(f"  全量样本: {len(df)} 行, {df.shape[1]} 列")

# 关键改动：Vina特征不填0，用NaN
# 特征合并脚本曾把无Vina的填了0，这里恢复成NaN
if "vina_available" in df.columns:
    for col in VINA_COLS:
        if col in df.columns:
            # vina_available=0的行，Vina特征设回NaN
            df.loc[df["vina_available"] == 0, col] = np.nan
    n_vina = int(df["vina_available"].sum())
    print(f"  Vina数据: {n_vina}/{len(df)} 样本有值, 其余设为NaN")
else:
    print(f"  [WARN] 无vina_available列")

# 排除CASF-285
casf_pdbs = set()
if CASF_CSV.exists():
    df_casf_list = pd.read_csv(CASF_CSV, sep="\t")
    casf_col = "#code" if "#code" in df_casf_list.columns else "code"
    casf_pdbs = set(df_casf_list[casf_col].str.lower().tolist())
    before = len(df)
    df_train = df[~df["pdb"].str.lower().isin(casf_pdbs)].reset_index(drop=True)
    df_casf_data = df[df["pdb"].str.lower().isin(casf_pdbs)].reset_index(drop=True)
    print(f"  排除CASF-285: {before} -> {len(df_train)} (train pool)")
    print(f"  CASF benchmark: {len(df_casf_data)} 样本")
else:
    df_train = df.copy()
    df_casf_data = pd.DataFrame()

# ============================================================
# 2. 特征工程
# ============================================================
print(f"\n[2/6] 特征工程")
meta_cols = {"pdb", "pKd", "split", "vina_available"}
feat_all = [c for c in df_train.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df_train[c])]

# 非Vina特征：inf->nan, fill 0
non_vina_feat = [c for c in feat_all if c not in VINA_COLS]
df_train[non_vina_feat] = df_train[non_vina_feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
# Vina特征：保留NaN，不填0！

print(f"  原始特征数: {len(feat_all)}")

# 方差过滤（只在非Vina特征上做，Vina特征不过滤）
from sklearn.feature_selection import VarianceThreshold
X_non_vina = df_train[non_vina_feat].values.astype(np.float32)
vt = VarianceThreshold(threshold=0.001)
vt.fit(X_non_vina)
feat_keep_non_vina = [f for f, keep in zip(non_vina_feat, vt.get_support()) if keep]
n_removed = len(non_vina_feat) - len(feat_keep_non_vina)
print(f"  方差过滤: 去除 {n_removed} 个低方差非Vina特征")
print(f"  保留非Vina: {len(feat_keep_non_vina)}, Vina: {len(VINA_COLS)}")
feat = feat_keep_non_vina + VINA_COLS
print(f"  总特征数: {len(feat)}")

# 分类统计
cat_counts = {}
for f in feat:
    if f.startswith("fp_morgan"): cat_counts["fp_morgan"] = cat_counts.get("fp_morgan", 0) + 1
    elif f.startswith("vina_"): cat_counts["vina"] = cat_counts.get("vina", 0) + 1
    elif f.startswith("desc_"): cat_counts["desc"] = cat_counts.get("desc", 0) + 1
    elif f.startswith(("apf_", "3d_", "hb_", "hydrophobic_", "saltbridge_", "contact_")): cat_counts["3d"] = cat_counts.get("3d", 0) + 1
    elif f.startswith(("prot_", "pocket_")): cat_counts["prot"] = cat_counts.get("prot", 0) + 1
    else: cat_counts["other"] = cat_counts.get("other", 0) + 1
print(f"  特征分类: {cat_counts}")

X = df_train[feat].values.astype(np.float32)  # XGBoost原生处理NaN
y = df_train["pKd"].values.astype(np.float32)
print(f"  训练样本: {len(df_train)}, 特征: {len(feat)}")
print(f"  NaN值总数: {int(np.isnan(X).sum())} (来自无Vina样本)")

# ============================================================
# 3. Optuna 超参搜索
# ============================================================
print(f"\n[3/6] Optuna {N_TRIALS} trials (max_depth 4-7, gap惩罚={GAP_PENALTY})")
import optuna
from xgboost import XGBRegressor
optuna.logging.set_verbosity(optuna.logging.WARNING)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
results = []

TRIALS_CSV = REPORTS_DIR / f"{MODEL_NAME}_optuna_trials.csv"
BEST_JSON = REPORTS_DIR / f"{MODEL_NAME}_best_params.json"

def objective(trial):
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        max_depth=trial.suggest_int("max_depth", 4, 7),  # 4-7
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 5, 30),
        reg_alpha=trial.suggest_float("reg_alpha", 0.5, 30.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    )
    val_r2, tr_r2 = [], []
    for tr, va in kf.split(X):
        m = XGBRegressor(**p)
        m.fit(X[tr], y[tr], verbose=False)
        val_r2.append(r2_score(y[va], m.predict(X[va])))
        tr_r2.append(r2_score(y[tr], m.predict(X[tr])))
    r2m = float(np.mean(val_r2))
    r2s = float(np.std(val_r2))
    trm = float(np.mean(tr_r2))
    gap = trm - r2m
    score = r2m - GAP_PENALTY * gap
    rec = dict(trial=trial.number, r2_mean=r2m, r2_std=r2s, r2_train_mean=trm,
               gap=gap, score=score,
               **{k: p[k] for k in ["n_estimators", "max_depth", "learning_rate",
                                    "subsample", "colsample_bytree",
                                    "min_child_weight", "reg_alpha", "reg_lambda"]})
    results.append(rec)
    pd.DataFrame(results).to_csv(TRIALS_CSV, index=False)
    best_score = max(r["score"] for r in results)
    print(f"  trial {trial.number:3d}: CV={r2m:.4f}+-{r2s:.4f} train={trm:.4f} gap={gap:.3f} "
          f"score={score:.4f} (best={best_score:.4f}) depth={p['max_depth']} n_est={p['n_estimators']}",
          flush=True)
    return score

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, gc_after_trial=True,
               show_progress_bar=False, catch=(Exception,))

# ============================================================
# 4. 选择最优超参
# ============================================================
print(f"\n[4/6] 选择最优超参")
rdf = pd.DataFrame(results)
best_by_score = rdf.sort_values("score", ascending=False).iloc[0]
best_by_r2 = rdf.sort_values("r2_mean", ascending=False).iloc[0]

print(f"\n  Top-5 by gap-penalized score:")
print(rdf.sort_values("score", ascending=False)[["trial", "r2_mean", "r2_train_mean",
      "gap", "score", "max_depth", "n_estimators"]].head(5).to_string(index=False))
print(f"\n  Top-5 by pure CV R2:")
print(rdf.sort_values("r2_mean", ascending=False)[["trial", "r2_mean", "r2_train_mean",
      "gap", "score", "max_depth", "n_estimators"]].head(5).to_string(index=False))

best = best_by_score
bp = {k: (int(best[k]) if k in ("n_estimators", "max_depth", "min_child_weight") else float(best[k]))
      for k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                "min_child_weight", "reg_alpha", "reg_lambda"]}
bp.update(objective="reg:squarederror", tree_method="hist",
          random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
print(f"\n  最优超参: {json.dumps(bp, indent=2)}")

# ============================================================
# 5. 最终CV + 训练 + 评估
# ============================================================
print(f"\n[5/6] 最终5折CV + 训练")
CV_FOLDS = REPORTS_DIR / f"{MODEL_NAME}_cv_folds.csv"
SPLIT_CSV = REPORTS_DIR / f"{MODEL_NAME}_split_metrics.csv"
TOP30 = REPORTS_DIR / f"{MODEL_NAME}_feature_importance_top30.csv"
BIAS_CSV = REPORTS_DIR / f"{MODEL_NAME}_bias_comparison.csv"
PERF = REPORTS_DIR / f"{MODEL_NAME}_performance.txt"
MODEL_PKL = MODELS_DIR / f"{MODEL_NAME}.pkl"

folds = []
for i, (tr, va) in enumerate(kf.split(X), 1):
    m = XGBRegressor(**bp)
    m.fit(X[tr], y[tr], verbose=False)
    me = dict(r2=r2_score(y[va], m.predict(X[va])),
              rmse=float(np.sqrt(mean_squared_error(y[va], m.predict(X[va])))),
              mae=mean_absolute_error(y[va], m.predict(X[va])))
    folds.append(dict(fold=i, n_train=len(tr), n_val=len(va), **me))
fdf = pd.DataFrame(folds)
fdf.to_csv(CV_FOLDS, index=False)
cv_r2 = fdf["r2"].mean()
cv_rmse = fdf["rmse"].mean()
cv_mae = fdf["mae"].mean()
print(f"  CV: R2={cv_r2:.4f} +- {fdf['r2'].std():.4f}  RMSE={cv_rmse:.4f}  MAE={cv_mae:.4f}")

m_final = XGBRegressor(**bp)
m_final.fit(X, y, verbose=False)
tr_pred = m_final.predict(X)
tr_r2 = r2_score(y, tr_pred)
tr_rmse = float(np.sqrt(mean_squared_error(y, tr_pred)))
tr_mae = mean_absolute_error(y, tr_pred)
gap = tr_r2 - cv_r2
print(f"  Train: R2={tr_r2:.4f}  gap={gap:.4f}")

sdf = pd.DataFrame([dict(split="train", r2=tr_r2, rmse=tr_rmse, mae=tr_mae),
                    dict(split="cv_mean", r2=cv_r2, rmse=cv_rmse, mae=cv_mae)])
sdf.to_csv(SPLIT_CSV, index=False)

# 偏置检测
lig_cols = [c for c in feat if c.startswith("desc_") or c.startswith("fp_morgan")]
Xb = df_train[lig_cols].values.astype(np.float32)
Xb = np.nan_to_num(Xb, nan=0.0)  # ligand-only无NaN
bias_cv = []
for tr, va in kf.split(Xb):
    mb = XGBRegressor(**bp)
    mb.fit(Xb[tr], y[tr], verbose=False)
    bias_cv.append(r2_score(y[va], mb.predict(Xb[va])))
bias_r2 = float(np.mean(bias_cv))
delta_r2 = cv_r2 - bias_r2
pd.DataFrame([dict(model="full", cv_r2=cv_r2, n_features=len(feat)),
              dict(model="ligand_only_desc_fp", cv_r2=bias_r2, n_features=len(lig_cols)),
              dict(model="delta", cv_r2=delta_r2, n_features=0)]).to_csv(BIAS_CSV, index=False)
print(f"  偏置: full={cv_r2:.4f}  ligand_only={bias_r2:.4f}  DeltaR2={delta_r2:+.4f}")

# 特征重要性
imp = m_final.feature_importances_
order = np.argsort(imp)[::-1][:30]
top30_df = pd.DataFrame([dict(rank=i+1, feature=feat[j], importance=float(imp[j]))
                         for i, j in enumerate(order)])
top30_df.to_csv(TOP30, index=False)
vina_in_top30 = [feat[j] for j in order if feat[j].startswith("vina_")]
print(f"  Vina特征在Top-30中: {vina_in_top30 if vina_in_top30 else '无'}")

cats = {}
for j, name in enumerate(feat):
    if name.startswith("desc_"): k = "desc"
    elif name.startswith("fp_morgan"): k = "fp_morgan"
    elif name.startswith("vina_"): k = "vina"
    elif name.startswith(("apf_", "3d_", "hb_", "hydrophobic_", "saltbridge_", "contact_")): k = "3d"
    elif name.startswith(("prot_", "pocket_")): k = "prot"
    else: k = "other"
    cats[k] = cats.get(k, 0) + float(imp[j])
tot = sum(cats.values()) + 1e-12
print(f"  特征重要性分类:")
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"    {k:12s} {v:.4f} ({v/tot*100:.1f}%)")

# ============================================================
# 6. CASF-2016 benchmark
# ============================================================
print(f"\n[6/6] CASF-2016 benchmark ({len(df_casf_data)}样本)")
CASF_CSV_OUT = REPORTS_DIR / f"{MODEL_NAME}_casf2016_benchmark.csv"
CASF_SUMMARY = REPORTS_DIR / f"{MODEL_NAME}_casf2016_summary.txt"

if len(df_casf_data) > 0:
    # CASF数据的Vina也用NaN
    for col in VINA_COLS:
        if col in df_casf_data.columns and "vina_available" in df_casf_data.columns:
            df_casf_data.loc[df_casf_data["vina_available"] == 0, col] = np.nan
    df_casf_data[non_vina_feat] = df_casf_data[non_vina_feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_casf = df_casf_data[feat].values.astype(np.float32)
    y_casf = df_casf_data["pKd"].values.astype(np.float32)
    casf_pred = m_final.predict(X_casf)

    r_pearson, p_p = pearsonr(y_casf, casf_pred)
    r_spearman, p_s = spearmanr(y_casf, casf_pred)
    r2_casf = r2_score(y_casf, casf_pred)
    rmse_casf = float(np.sqrt(mean_squared_error(y_casf, casf_pred)))
    mae_casf = mean_absolute_error(y_casf, casf_pred)

    casf_df = pd.DataFrame({
        "pdb": df_casf_data["pdb"].values,
        "true_pKd": y_casf,
        "pred_pKd": casf_pred,
        "error": casf_pred - y_casf,
    })
    casf_df.to_csv(CASF_CSV_OUT, index=False)

    summary = f"""CASF-2016 core-set ({len(df_casf_data)}) external benchmark (Step4 v3)
========================================================
训练样本: {len(df_train)} (全量, 排除CASF-285)
Vina处理: NaN不填0 (XGBoost原生缺失值处理)
特征数: {len(feat)} (方差过滤后)
模型: XGBoost (gap-penalized Optuna, max_depth 4-7)

Pearson R  = {r_pearson:.4f}  (p={p_p:.2e})
R^2        = {r2_casf:.4f}
Spearman   = {r_spearman:.4f}
RMSE       = {rmse_casf:.4f} pKd
MAE        = {mae_casf:.4f} pKd

文献对标:
  DeltaVinaXGB (2019):  Pearson R = 0.828
  DeltaLinF9XGB (2022): Pearson R = 0.853
  RF-Score v1:           Pearson R ~ 0.65
  baseline v1 (14055,填0):  R = 0.8234
  v2 (1612, 只用Vina):   R = 0.7171
  v3 (本次, 14055+NaN):  R = {r_pearson:.4f}
"""
    CASF_SUMMARY.write_text(summary, encoding="utf-8")
    print(summary)
else:
    print("  [WARN] 无CASF数据")

# 保存
with open(MODEL_PKL, "wb") as fh:
    pickle.dump(dict(model=m_final, feature_cols=feat, params=bp), fh)
with open(BEST_JSON, "w", encoding="utf-8") as fh:
    json.dump(dict(best_trial=int(best["trial"]), best_score=float(best["score"]),
                   best_cv_r2=float(best["r2_mean"]), best_cv_r2_std=float(best["r2_std"]),
                   best_train_r2=float(best["r2_train_mean"]), best_gap=float(best["gap"]),
                   params=bp), fh, indent=2, ensure_ascii=False)
with open(PERF, "w", encoding="utf-8") as fh:
    fh.write(f"model={MODEL_NAME}\n")
    fh.write(f"samples_train={len(df_train)} features={len(feat)}\n")
    fh.write(f"vina_strategy=NaN_not_zero\n")
    fh.write(f"CV_R2={cv_r2:.4f} CV_R2_std={fdf['r2'].std():.4f} CV_RMSE={cv_rmse:.4f} CV_MAE={cv_mae:.4f}\n")
    fh.write(f"TRAIN_R2={tr_r2:.4f}\n")
    fh.write(f"GAP_train_cv={gap:.4f}\n")
    fh.write(f"BIAS_ligand_only_R2={bias_r2:.4f} DELTA_R2={delta_r2:+.4f}\n")
    fh.write(f"VINA_TOP30={vina_in_top30}\n")
    if len(df_casf_data) > 0:
        fh.write(f"CASF2016_R={r_pearson:.4f} CASF2016_R2={r2_casf:.4f} CASF2016_RMSE={rmse_casf:.4f}\n")
    fh.write(f"best_params={json.dumps(bp, ensure_ascii=False)}\n")

print(f"\n{'='*70}")
print(f"[完成] 模型: {MODEL_PKL.name}")
print(f"  CV R2 = {cv_r2:.4f} (gap={gap:.4f})")
print(f"  偏置 DeltaR2 = {delta_r2:+.4f}")
if len(df_casf_data) > 0:
    print(f"  CASF-2016 R = {r_pearson:.4f}")
print(f"{'='*70}")
