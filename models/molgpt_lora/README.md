# MolGPT-LoRA — 靶点配体空间定向分子生成（参数高效微调适配层）

本目录包含基于 MolGPT 的 LoRA 微调适配层权重与完整的训练 / 生成 / 质检代码。
用途：用某个靶点蛋白的少量**公开已知配体**（数十条量级）对预训练分子生成模型做
参数高效微调，使生成分布向该靶点的配体化学空间偏移；产出的生成分子随后与普通
库分子一同进入**统一的质检与打分流程**（见 §4），不享有任何"生成式优待"。

**基座模型不随本仓库分发**：请从 HuggingFace 拉取 `msb-roshan/molgpt`
（GPT-2 LM head 架构，参数量约 108.9M）。
MolGPT 出处：Bagal V, Aggarwal R, Vinod PK, Priyakumar UD. *MolGPT: Molecular
Generation Using a Transformer-Decoder Language Model.* J. Chem. Inf. Model.
2022, 62(9), 2046–2059.

## 1. 目录内容

| 文件 | 说明 |
|---|---|
| `adapter/adapter_model.safetensors` | LoRA 权重（约 1.18 MB，不含基座） |
| `adapter/adapter_config.json` | PEFT 配置 |
| `adapter/tokenizer.json` / `adapter/tokenizer_config.json` | tokenizer |
| `adapter/train_log.json` | 训练事实记录（loss / epochs / 数据规模等原文） |
| `code/` | 数据集构建 / 微调 / 生成 / 质检脚本（见 §4、§5） |

## 2. 训练事实（取自 `adapter/train_log.json` 与脚本常量，未改写）

- 任务：CLM 自回归 SMILES 生成，输入序列 = `[BOS] + SMILES tokens + [EOS]`
- LoRA：r=8，lora_alpha=32，dropout=0.1，`target_modules=["c_attn"]`（GPT-2 的 QKV 合并层）
- 训练：10 epochs，lr=3e-4，batch_size=4 × 梯度累积 4，GPU 上 fp16 自动开启
- 可训练参数占比 **0.48%**（<5%，属于参数高效微调而非全量再训练）
- round-trip 校验：训练集全部 SMILES 编码-解码无损
- 环境实测：PEFT 0.20.0；`transformers>=4.46`（Trainer 的 `processing_class` 参数）

## 3. 训练数据来源与防泄漏设计

- **数据**：目标靶点的公开已知配体 SMILES，数十条量级（发布版 adapter 为 37 条），
  由 `code/build_finetune_dataset.py` 从外部收集的配体表构建（结构合法性校验、
  canonical 化、去重）。
- **阳性对照断言剔除（防泄漏关键）**：若配体表中存在项目指定的阳性对照分子，
  构建脚本会在 **canonical SMILES / 去立体 SMILES / Source 标签** 三条路径上
  独立校验并强制剔除——任一路命中即中止写出，防止标签笔误绕过字符串匹配；
  被剔除的分子单独备份为 CSV，供生成后"命中证据"对照使用
  （`--control-name` / `--control-backup` 参数）。
- 保留种子分子与阳性对照的 Tanimoto 同族/异族检查，用于确认训练集化学空间代表性。

## 4. 生成分子质检流程（code/ 五个脚本）

1. **`generate_and_compare.py`**：加载基座（+可选 adapter）批量采样生成 SMILES，
   RDKit 解析过滤 + canonical 去重；计算与训练集/阳性对照的 Tanimoto 相似度，
   输出对比图。生成参数：`max_length=BOS+32`，top_k=96，top_p=0.92，
   temperature=0.88，repetition_penalty=1.12。
2. **`qc_filter_merge.py` — 四道关卡过滤**：Lipinski（MW<500/LogP<5/HBD≤5/HBA≤10）
   → Veber（TPSA≤140/RotB≤10）→ PAINS（rdkit FilterCatalog）→ SA Score（Ertl，
   <4.5 易合成）；通过者与既有参考库 canonical 去重后并入生成分子库（ID 前缀 `GEN_`）。
3. **`qc_sample.py` — 分层抽样质检**：按与训练集相似度三分位分层（高/中/低）抽样，
   自动结构风险检测（长脂肪链/O-O 键/酰氯/多硝基/环数），RDKit 逐条出 PNG +
   PIL 拼总览网格，供人工审查。
4. **`qc_structural_review.py` — 结构复审**：对抽样逐条基于真实 SMILES 重算描述符
   （MW/cLogP/TPSA/Fsp3/环系/最长链）+ PAINS/Brenk 警报 + 异常原子
   （形式电荷/类金属），修正抽样表的机器预判。
5. **`qc_final_clean.py` — 机械规则全量清理**（硬踢）：
   R1 类金属/重金属原子；R2 游离磷（P 邻居无 O）；R3 负电 C/N（含异腈）；
   R4 游离醇盐（N-oxide 与盐形式白名单保留）；R5 净电荷 |q|>1；R6 MW<150 碎片。
   产出最终保留库与剔除清单。**标注（不踢）**：与阳性对照 Tanimoto>0.85 的分子
   仅作"命中证据"标注（高相似是模型学到该配体化学空间的正向信号，而非数据污染）；
   与训练集 Tc≥0.999 标注为记忆复现。

## 5. 如何加载使用

### 5.1 命令行（推荐走脚本）

```bash
cd code

# 1) 构建微调数据集（含阳性对照断言剔除）
python build_finetune_dataset.py --input ligands.csv --output finetune_smiles.csv \
    --control-name positive_control --control-backup control_backup.csv

# 2) LoRA 微调
python finetune_molgpt_lora.py --data finetune_smiles.csv \
    --base-model msb-roshan/molgpt --out molgpt_lora_adapter

# 3) 生成与对比（--adapter 不传则只跑基线组）
python generate_and_compare.py --data finetune_smiles.csv \
    --base-model msb-roshan/molgpt --adapter ../adapter \
    --control-backup control_backup.csv --out-dir output/gen --n-gen 1000

# 4) 质检链
python qc_filter_merge.py --gen-smi output/gen/finetuned.smi \
    --source-lib reference_library.csv --out-csv generated_library_merged.csv
python qc_sample.py --train-csv finetune_smiles.csv --lib-csv generated_library_merged.csv \
    --out-dir output/qc
python qc_structural_review.py --in-csv output/qc/qc_sample_24.csv \
    --out-csv output/qc/structural_review.csv
python qc_final_clean.py --input generated_library_merged.csv \
    --control-backup control_backup.csv --train-csv finetune_smiles.csv \
    --out-keep generated_library_final.csv --out-rej rejected_molecules.csv
```

### 5.2 Python 直接加载 adapter 生成

```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像可选
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from peft import PeftModel

BASE = "msb-roshan/molgpt"
tok = AutoTokenizer.from_pretrained(BASE)
model = GPT2LMHeadModel.from_pretrained(BASE)
model.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(model, "adapter")   # 挂载 LoRA 适配层
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 与 code/generate_and_compare.py 同参：BOS 前缀、受控采样
input_ids = torch.tensor([[tok.bos_token_id]], device=model.device)
out = model.generate(
    input_ids,
    max_length=input_ids.size(1) + 32,
    do_sample=True, top_k=96, top_p=0.92, temperature=0.88,
    repetition_penalty=1.12, num_return_sequences=8,
    pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
)
for seq in out:
    smi = tok.decode(seq.tolist(), skip_special_tokens=True)
    print(smi)
```

> 生成结果务必先过 RDKit 解析过滤与 §4 质检链，再进入下游评估。

## 6. 局限（如实披露）

- 微调集仅数十条，是靶点已知配体稀缺的客观限制；生成多样性与合成可及性受限。
- 生成分子只是**候选来源之一**，必须与普通库分子走完全相同的质检与打分门控；
  本目录质检脚本只做结构合理性与类药性质把关，结合亲和力评估需另行进行
  （可用本仓库 `models/affinx` 或分子对接工具）。
- 受控采样下仍可能产出化学上不合理结构；人工抽样审查（qc_sample →
  qc_structural_review）不可省略。
- 训练 loss 绝对值在小数据下参考意义有限，评估重点应放在生成分布、
  round-trip 无损率与质检通过率上。

## 7. 引用

- MolGPT（基座）：Bagal V. et al., *J. Chem. Inf. Model.* 2022, 62, 2046–2059
- LoRA：Hu E. et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- PEFT：HuggingFace `peft` 库
- 质检阈值参考：Lipinski CA. *Rule of five in 2000*; Veber DF. et al. 2002;
  Baell JB, Holloway GA. *New Substructure Filters for Removal of Pan Assay
  Interference Compounds (PAINS) from Screening Libraries.* J. Med. Chem. 2010;
  Ertl P., Schuffenhauer A. *Estimation of synthetic accessibility score of
  drug-like molecules based on molecular complexity and fragment contributions.*
  J. Cheminform. 2009, 1, 8.

## 8. 免责声明

本模型生成的分子仅为**计算假设**，不构成任何成药结论；所有生成分子在通过完整
质检链与独立结合评估之前，不应视为活性候选。
