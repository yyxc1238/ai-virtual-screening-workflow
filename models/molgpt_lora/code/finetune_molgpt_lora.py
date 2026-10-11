# -*- coding: utf-8 -*-
"""
MolGPT(GPT2LMHead) LoRA轻量级微调 —— 因果语言建模(CLM)
=====================================================================
数据：靶点已知配体 SMILES（几十条量级，见 build_finetune_dataset.py 产出）
原理：
  - MolGPT为自回归生成模型，训练目标是给定前缀预测下一个token
  - 输入序列 = [BOS] + SMILES tokens + [EOS]
  - LoRA只训练attention适配层（<5%参数）→ 实质性优化
产出：molgpt_lora_adapter/（adapter 权重 + tokenizer + train_log.json）

用法：
  python finetune_molgpt_lora.py --data finetune_smiles.csv \
      --base-model msb-roshan/molgpt --out ../adapter
  （--base-model 传 HuggingFace 模型 ID 或本地已下载的模型目录均可）
"""
import os
import sys
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

MAX_LEN = 64            # 模型n_ctx=64；小分子SMILES实测远短于此
EPOCHS = 10             # 小数据CLM需要稍多轮次；配合dropout防过拟合
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 3e-4               # LoRA典型学习率
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


class SmilesCLMDataset(Dataset):
    """[BOS]+SMILES+[EOS] 因果语言建模数据集"""

    def __init__(self, smiles_list, tokenizer, max_len):
        self.examples = []
        for smi in smiles_list:
            ids = tokenizer(smi, add_special_tokens=False).input_ids
            ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            ids = ids[:max_len]
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        labels = input_ids.clone()  # CLM: 标签=输入(内部自动shift)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate_pad(batch, pad_id):
    """按批内最长序列padding"""
    maxlen = max(x["input_ids"].size(0) for x in batch)
    out = {}
    for k in ("input_ids", "labels", "attention_mask"):
        padded = torch.full((len(batch), maxlen), (pad_id if k != "attention_mask" else 0),
                            dtype=torch.long)
        for i, x in enumerate(batch):
            seq = x[k]
            if k == "labels":
                pad_labels = torch.full_like(seq, -100)  # padding位不计loss
                pad_labels[:seq.size(0)] = seq
                padded[i, :seq.size(0)] = pad_labels
            else:
                padded[i, :seq.size(0)] = seq
        out[k] = padded
    return out


def main():
    ap = argparse.ArgumentParser(description="MolGPT LoRA 微调（CLM自回归生成任务）")
    ap.add_argument("--data", required=True, help="训练集 CSV（SMILES 列）")
    ap.add_argument("--base-model", default="msb-roshan/molgpt",
                    help="基座模型（HuggingFace ID 或本地目录）")
    ap.add_argument("--out", default="molgpt_lora_adapter", help="LoRA 权重输出目录")
    ap.add_argument("--log-dir", default="output/logs", help="训练日志/检查点目录")
    args = ap.parse_args()

    print("=" * 60)
    print("MolGPT LoRA微调（CLM自回归生成任务）")
    print("=" * 60)
    if not os.path.exists(args.data):
        print(f"[ERROR] 缺文件: {args.data}")
        sys.exit(1)

    print("[1] 加载基座MolGPT")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = GPT2LMHeadModel.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK 参数量={n_params/1e6:.1f}M")

    print("[2] 准备CLM数据集")
    df = pd.read_csv(args.data)
    dataset = SmilesCLMDataset(df["SMILES"].tolist(), tokenizer, MAX_LEN)
    lens = [len(e) for e in dataset.examples]
    print(f"  样本数={len(dataset)}, 序列长度 min={min(lens)} max={max(lens)}")

    print("[3] 配置LoRA（GPT2 attention打包层 c_attn）")
    cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn"],   # GPT2的QKV在c_attn线性层
        bias="none",
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable/1e6:.3f}M / {total/1e6:.1f}M ({trainable/total*100:.2f}%)")

    print("[4] 训练")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    targs = TrainingArguments(
        output_dir=os.path.join(args.log_dir, "molgpt_ckpt"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        logging_steps=2,
        save_strategy="no",           # 小任务直接存最终LoRA即可
        fp16=torch.cuda.is_available(),
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=lambda b: collate_pad(b, tokenizer.pad_token_id or tokenizer.eos_token_id),
        processing_class=tokenizer,
    )
    res = trainer.train()
    print(f"  训练loss: {res.training_loss:.4f}")

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    log = {
        "model": args.base_model,
        "task": "CLM autoregressive generation",
        "train_loss": res.training_loss,
        "epochs": EPOCHS,
        "lr": LR,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "dataset_size": len(dataset),
        "gpu_used": torch.cuda.is_available(),
    }
    with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n完成！LoRA权重: {args.out}")


if __name__ == "__main__":
    main()
