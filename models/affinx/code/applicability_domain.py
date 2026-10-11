# -*- coding: utf-8 -*-
"""
applicability_domain.py —— XGBoost pKd 模型的适用域（AD）与预测不确定性
=====================================================================
设计原则：
  1. 树模型本身不给出置信区间。化学计量学中对 QSAR 模型最常用、最透明的
     适用域判据是"与训练集最近邻的结构相似度"：若查询分子与任何训练分子
     都不像，则预测来自外推，不可与域内预测同等信任。
  2. 结构表征与模型输入完全一致：2,048 bit Morgan 指纹（半径 2，RDKit
     MorganGenerator，含氢分子），相似度用 Tanimoto（Jaccard）：
         T(q, t) = |bit_q ∩ bit_t| / |bit_q ∪ bit_t|
  3. 参考库 = 模型实际训练池 13,770 条 PDBBind v2020 复合物的配体指纹
     （全量特征表 14,055 条排除 CASF-2016 285 个外部基准；与 2,110 维特征
     中的 fp_morgan_0..2047 同源，由 build_train_fp_ref.py 从全量特征表
     抽取打包，即随仓库发布的 train_fingerprints_13770.npz）。
  4. 不确定性区间：以五折交叉验证残差 RMSE=1.2318 pKd 为基准 sigma0
     （发布模型训练日志原文），再按最近邻距离做透明的线性放大（倍数 1→2），
     不做未经数据校准的分布假设；95% 区间取 ±1.96·sigma（高斯残差近似，
     已在 README 中注明这是工程化近似而非保形校准区间）。

阈值（事前固定、写入 README，不随结果回调）：
  max Tanimoto ≥ 0.50  IN_DOMAIN       sigma = 1.00 × CV_RMSE
  0.30 ≤ T < 0.50      BORDERLINE      sigma 在 1.0–1.5×CV_RMSE 间线性放大
  T < 0.30             OUT_OF_DOMAIN   sigma 在 1.5–2.0×CV_RMSE 间线性放大
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# 与发布模型训练日志一致
CV_RMSE_PKD = 1.2318

SIM_IN_DOMAIN = 0.50
SIM_OUT_OF_DOMAIN = 0.30

# 参考库与本文件同在 models/affinx/ 下（本文件位于 models/affinx/code/）
FP_REF_PATH = (Path(__file__).resolve().parents[1]
               / "train_fingerprints_13770.npz")


def morgan_bits(smiles: str):
    """SMILES -> 2,048 维 uint8 Morgan 指纹（与 predict.compute_2d_features 同口径）。"""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molH = Chem.AddHs(mol)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS,
                                                    fpSize=MORGAN_NBITS)
    arr = np.zeros(MORGAN_NBITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(molH), arr)
    return arr.astype(np.uint8)


class FpReference:
    """训练池指纹参考库（packbits 存储，13,770×256 字节 ≈ 3.5 MB / 压缩后 0.8 MB）。"""

    def __init__(self, path: Path | str = FP_REF_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"训练指纹参考库不存在: {path}\n"
                "请先运行 build_train_fp_ref.py"
                "（需全量特征表 features_3d_vina_5316.csv，获取方式见 README.md）。")
        z = np.load(path, allow_pickle=True)
        self.ids = z["pdb"].astype(str)
        self.pkd = z["pkd"].astype(np.float32)
        self.split = z["split"].astype(str)
        # (N,256) 每字节压缩 8 个指纹位，用于绕开 BLAS 的 popcount 相似度
        self.bits_packed = z["bits_packed"].astype(np.uint8)
        self.bits = np.unpackbits(self.bits_packed, axis=1)[:, :MORGAN_NBITS] \
            .astype(np.uint8)
        self.popcount = self.bits.sum(axis=1).astype(np.int32)
        assert self.bits.shape[1] == MORGAN_NBITS

    def nearest_neighbor(self, qbits: np.ndarray):
        """返回 (max_tanimoto, nearest_pdb_id, nearest_pKd, nearest_split)。"""
        sim = self._batch_similarity(qbits[None, :])[0]
        i = int(np.argmax(sim))
        return float(sim[i]), str(self.ids[i]), float(self.pkd[i]), str(self.split[i])

    def _batch_similarity(self, qbits: np.ndarray, chunk: int = 32) -> np.ndarray:
        """M 个查询指纹 vs N 条参考指纹的 Tanimoto 矩阵 (M, N)。

        实现说明（环境适配，勿改成 float 矩阵乘）：
          交集大小 = 对两侧 packbits 字节做 AND 后逐字节 popcount 求和，
          使用 numpy≥2.0 的 np.bitwise_count，全程整数运算。
          原因：部分 conda OpenBLAS 环境在 float32 sgemm 大矩阵上会触发
          访问冲突（0xC0000005，进程直接崩溃，OPENBLAS_NUM_THREADS=1 亦复现），
          故不使用 BLAS 点积。按 32 个查询分块，临时 AND/popcount 数组约
          110 MB，数百分子秒级完成。
        """
        if not hasattr(np, "bitwise_count"):
            raise RuntimeError(
                "np.bitwise_count 需要 numpy>=2.0；当前 numpy 版本过低，"
                "请升级或改用参考库最近邻的第三方实现。")
        q = qbits.astype(np.uint8)
        qp = np.packbits(q, axis=1)                       # (M,256)
        q_pop = q.sum(axis=1).astype(np.int32)
        rp = self.bits_packed                             # (N,256)
        r_pop = self.popcount
        sims = np.empty((q.shape[0], rp.shape[0]), dtype=np.float32)
        for s0 in range(0, q.shape[0], chunk):
            qc = qp[s0:s0 + chunk]                        # (m,256)
            and_bytes = qc[:, None, :] & rp[None, :, :]   # (m,N,256)
            inter = np.bitwise_count(and_bytes).sum(axis=2, dtype=np.int32)
            union = r_pop[None, :] + q_pop[s0:s0 + qc.shape[0], None] - inter
            sims[s0:s0 + qc.shape[0]] = inter / np.maximum(union, 1)
        return sims

    def nearest_neighbor_batch(self, qbits: np.ndarray):
        """返回 (max_sim(M,), nearest_idx(M,))；idx 用于取 ids/pkd/split。"""
        sim = self._batch_similarity(qbits)
        idx = np.argmax(sim, axis=1)
        return sim[np.arange(len(idx)), idx], idx


def domain_flag(max_sim: float) -> str:
    if max_sim >= SIM_IN_DOMAIN:
        return "IN_DOMAIN"
    if max_sim >= SIM_OUT_OF_DOMAIN:
        return "BORDERLINE"
    return "OUT_OF_DOMAIN"


def uncertainty_sigma(max_sim: float) -> float:
    """按最近邻相似度放大的预测标准差（pKd 单位）。"""
    if max_sim >= SIM_IN_DOMAIN:
        factor = 1.0
    elif max_sim >= SIM_OUT_OF_DOMAIN:
        # 0.50→1.0, 0.30→1.5 线性
        factor = 1.0 + (SIM_IN_DOMAIN - max_sim) / (
            SIM_IN_DOMAIN - SIM_OUT_OF_DOMAIN) * 0.5
    else:
        # 0.30→1.5, 0.00→2.0 线性
        factor = 1.5 + (SIM_OUT_OF_DOMAIN - max_sim) / SIM_OUT_OF_DOMAIN * 0.5
    return CV_RMSE_PKD * factor


def assess(smiles: str, ref: FpReference):
    """一条 SMILES 的完整 AD+不确定性评估。失败返回 None。"""
    q = morgan_bits(smiles)
    if q is None:
        return None
    sim, nn_id, nn_pkd, nn_split = ref.nearest_neighbor(q)
    sigma = uncertainty_sigma(sim)
    return {
        "domain_max_tanimoto": round(sim, 4),
        "domain_nearest_train_pdb": nn_id,
        "domain_flag": domain_flag(sim),
        "sigma_pKd": round(sigma, 4),
    }
