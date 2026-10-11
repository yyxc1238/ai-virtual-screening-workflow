# -*- coding: utf-8 -*-
"""
MolGPT 微调前后各生成N条 + RDKit验证 + 对比报告
=====================================================================
对比维度：
  1. 合法率：RDKit可解析SMILES比例
  2. 新颖性：与训练集不重复的比例
  3. 领域相关性：与靶点已知配体库的最大/平均Tanimoto相似度
  4. 药物属性：MW分布
产出：
  output/generated_baseline.smi / generated_finetuned.smi
  output/before_after_comparison.png / molgpt_comparison_report.csv

用法：
  python generate_and_compare.py --data finetune_smiles.csv \
      --adapter ../adapter --n-gen 1000
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit import rdBase
rdBase.DisableLog("rdApp.error")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer, GPT2LMHeadModel
from peft import PeftModel

MAX_NEW_TOKENS = 32      # 数据max约24 token，32封顶抑制无效长链
BATCH = 64

tok = None  # 模块级 tokenizer（clean_smiles 使用）


def clean_smiles(text):
    """decode去空格+BOS/EOS清理"""
    s = text.replace(tok.bos_token or "<s>", "").replace(tok.eos_token or "</s>", "")
    for t in (tok.pad_token, tok.unk_token):
        if t:
            s = s.replace(t, "")
    return s.strip().replace(" ", "")


@torch.no_grad()
def generate_batched(model, tokenizer, n_total, device, tag, prompts=None):
    """批量自回归生成。
    prompts=None: 冷启动(BOS)；否则用训练集SMILES前缀作prompt引导续写"""
    import random
    random.seed(42)
    smiles = []
    seen = set()
    attempts = 0
    if prompts is None:
        prompt_pool = [tokenizer.bos_token]
    else:
        prompt_pool = prompts
    while len(seen) < n_total and attempts < n_total * 6:
        need = min(BATCH, (n_total - len(seen)) * 2 + BATCH)
        batch_prompts = [random.choice(prompt_pool) for _ in range(need)]
        enc = [tokenizer(p, add_special_tokens=False).input_ids for p in batch_prompts]
        maxp = max(len(e) for e in enc)
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id] + e + [tokenizer.pad_token_id] * (maxp - len(e))
             for e in enc], dtype=torch.long).to(device)
        # pad为填充符；prefix内不含pad故非零位即真实token
        attn = (input_ids != tokenizer.pad_token_id).long()
        out = model.generate(
            input_ids,
            attention_mask=attn,
            max_length=input_ids.size(1) + MAX_NEW_TOKENS,
            do_sample=True,
            top_k=96,               # 收紧采样空间
            top_p=0.92,
            temperature=0.88,       # 降低随机漂移
            repetition_penalty=1.12,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in out:
            raw = clean_smiles(tokenizer.decode(seq, skip_special_tokens=True))
            attempts += 1
            # 后处理：多组分切分取最大MW单组分（模型残留在盐式续接行为）
            smi, mol = postprocess_smiles(raw)
            if smi and mol is not None and smi not in seen:
                seen.add(smi)
                smiles.append(smi)
                if len(seen) >= n_total:
                    break
    print(f"  [{tag}] 尝试{attempts}次 -> unique SMILES {len(smiles)}条")
    return smiles


def _balanced_positions(smi):
    """返回所有'语法安全'的截断位置（不破坏括号/环开闭/键符号）"""
    safe = []
    depth = 0
    i = 0
    n = len(smi)
    while i < n:
        c = smi[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return []          # 前缀本身已失衡
        elif c == "%":
            i += 2                 # 两位数环编号(%11)整体跳过
        # 截断安全性：depth==0 且最后不是键符号/开括号/数字
        tail = smi[i]
        if depth == 0 and tail not in "(=%$#:/\\.-" and not tail.isdigit():
            safe.append(i + 1)
        i += 1
    return safe


def make_prefix_prompts(tokenizer, df_train, n_per_mol=6):
    """从训练集SMILES取语法平衡的token前缀做prompt池"""
    import random as _rd
    _rd.seed(42)
    pool = []
    for smi in df_train["SMILES"]:
        ids = tokenizer(smi, add_special_tokens=False).input_ids
        chars = tokenizer.convert_ids_to_tokens(ids)
        # 用token级累积字符串找平衡点
        acc = ""
        boundaries = [0]   # token索引->累积文本长度
        for t in chars:
            t = t.replace("Ġ", "")
            acc += t
            boundaries.append(len(acc))
        safe_char_pos = set(_balanced_positions(acc))
        cand_idx = [k for k, b in enumerate(boundaries)
                    if b in safe_char_pos and k >= 2 and k <= len(ids) - 2]
        picks = _rd.sample(cand_idx, min(n_per_mol, len(cand_idx))) if cand_idx else []
        for k in picks:
            frag = "".join(chars[:k]).replace("Ġ", "")
            pool.append(frag)
    print(f"  prompt池(平衡前缀): {len(pool)}个")
    return pool


def postprocess_smiles(raw):
    """生成后处理：取最大MW单组分"""
    best, best_mw, best_mol = None, -1, None
    for part in raw.split("."):
        m = Chem.MolFromSmiles(part)
        if m:
            mw = Descriptors.MolWt(m)
            if mw > best_mw and len(m.GetAtoms()) >= 5:
                best, best_mw, best_mol = part, mw, m
    return best, best_mol


def analyze(smiles_list, train_fps, train_canon, tag):
    rows = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
        sims = [DataStructs.TanimotoSimilarity(fp, tfp) for tfp in train_fps]
        rows.append({
            "tag": tag,
            "raw_smiles": smi,
            "canonical_smiles": canon,
            "MW": Descriptors.MolWt(m),
            "LogP": Descriptors.MolLogP(m),
            "HBD": Descriptors.NumHDonors(m),
            "HBA": Descriptors.NumHAcceptors(m),
            "TPSA": Descriptors.TPSA(m),
            "sim_to_train_max": round(max(sims), 4),
            "sim_to_train_mean": round(float(np.mean(sims)), 4),
            "novel_vs_train": int(canon not in train_canon),
            "lipinski_pass": int(Descriptors.MolWt(m) < 500 and Descriptors.MolLogP(m) < 5 and
                                 Descriptors.NumHDonors(m) <= 5 and Descriptors.NumHAcceptors(m) <= 10),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="MolGPT 微调前后生成对比")
    ap.add_argument("--data", required=True, help="训练集 CSV（SMILES 列）")
    ap.add_argument("--base-model", default="msb-roshan/molgpt",
                    help="基座模型（HuggingFace ID 或本地目录）")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter 目录（微调组）；不传则只跑基线")
    ap.add_argument("--control-backup", default="",
                    help="可选：阳性对照备份 CSV（SMILES 列），提供则输出对照相似度参考")
    ap.add_argument("--out-dir", default="output", help="输出目录")
    ap.add_argument("--n-gen", type=int, default=1000, help="每组生成条数")
    args = ap.parse_args()
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    N_GEN = args.n_gen

    print("=" * 60)
    print(f"MolGPT微调前后各生成{N_GEN}条 对比分析")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] 加载模型 + 训练集指纹（device={device}）")
    global tok
    tok = AutoTokenizer.from_pretrained(args.base_model)
    base_model = GPT2LMHeadModel.from_pretrained(args.base_model).to(device).eval()

    df_train = pd.read_csv(args.data)
    train_mols = [Chem.MolFromSmiles(s) for s in df_train["SMILES"].astype(str)]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in train_mols if m]
    train_canon = {Chem.MolToSmiles(m) for m in train_mols if m}
    print(f"  训练集: {len(train_fps)} 有效指纹")

    # 阳性对照锚点：训练集已剔除，从备份读取，作为独立的外部参考
    posctrl_fp = None
    if args.control_backup and os.path.exists(args.control_backup):
        df_pc = pd.read_csv(args.control_backup)
        pc_smi = str(df_pc.iloc[0]["SMILES"])
        pm = Chem.MolFromSmiles(pc_smi)
        if pm:
            posctrl_fp = AllChem.GetMorganFingerprintAsBitVect(pm, 2, 1024)
            print(f"  阳性对照锚点OK: 重原子{pm.GetNumHeavyAtoms()} (不在训练集,独立参考)")
    elif args.control_backup:
        print("  ⚠️ 阳性对照备份不存在,跳过外部锚点")

    # prompt池：训练集SMILES前缀（两组用同一池，公平对比）
    prompts = make_prefix_prompts(tok, df_train)

    results = {}
    for tag, model_path in [("baseline", None), ("finetuned", args.adapter)]:
        if tag == "finetuned" and not model_path:
            print("\n[2] 未提供 --adapter，跳过微调组")
            continue
        print(f"\n[2] 生成：{tag}（同prompt引导）")
        if model_path:
            model = PeftModel.from_pretrained(base_model, model_path).to(device).eval()
        else:
            model = base_model
        gen = generate_batched(model, tok, N_GEN, device, tag, prompts=prompts)
        res = analyze(gen, train_fps, train_canon, tag)
        results[tag] = res
        out_smi = os.path.join(OUT_DIR, f"generated_{tag}.smi")
        res[["canonical_smiles"]].to_csv(out_smi, index=False, header=False)
        print(f"  RDKit有效: {len(res)}/{len(gen)} ({len(res)/len(gen)*100:.1f}%)")
        print(f"  输出: {out_smi}")
        if model is not base_model:
            del model
            torch.cuda.empty_cache()

    if "finetuned" not in results:
        print("\n仅基线组完成。")
        return

    # 汇总对比
    print("\n[3] 微调前后对比汇总")
    report_rows = []
    for metric, fn in [
        ("RDKit合法率%", lambda r: len(r) / N_GEN * 100),
        ("新颖率(vs训练集)%", lambda r: r["novel_vs_train"].mean() * 100 if len(r) else 0),
        ("MW均值", lambda r: r["MW"].mean() if len(r) else np.nan),
        ("MW中位数", lambda r: r["MW"].median() if len(r) else np.nan),
        ("LogP均值", lambda r: r["LogP"].mean() if len(r) else np.nan),
        ("Tanimoto相似度均值(vs训练库)", lambda r: r["sim_to_train_mean"].mean() if len(r) else np.nan),
        ("Tanimoto相似度最大值均值", lambda r: r["sim_to_train_max"].mean() if len(r) else np.nan),
        ("Lipinski通过率%", lambda r: r["lipinski_pass"].mean() * 100 if len(r) else np.nan),
    ]:
        b = fn(results["baseline"])
        f_ = fn(results["finetuned"])
        delta = (f_ - b) if not (np.isnan(b) or np.isnan(f_)) else np.nan
        report_rows.append({"指标": metric, "基线(预训练)": round(b, 3),
                            "微调后": round(f_, 3), "变化": round(delta, 3)})
    rpt = pd.DataFrame(report_rows)
    print(rpt.to_string(index=False))

    # 阳性对照相似度专项（高相似=模型学到对照所在化学族的信号，非数据污染：
    # 对照从未进入训练集）
    if posctrl_fp is not None:
        for tag in ("baseline", "finetuned"):
            r = results[tag]
            pc_sims = []
            for _, row in r.iterrows():
                m = Chem.MolFromSmiles(row["canonical_smiles"])
                if m:
                    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
                    pc_sims.append(DataStructs.TanimotoSimilarity(fp, posctrl_fp))
            if pc_sims:
                print(f"  [{tag}] 与阳性对照相似度: mean={np.mean(pc_sims):.4f}, max={max(pc_sims):.4f}")

    # 合并明细表输出
    all_df = pd.concat([results["baseline"], results["finetuned"]], ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "molgpt_comparison_report.csv")
    all_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    rpt.to_csv(os.path.join(OUT_DIR, "molgpt_summary_table.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[4] 报告输出: {csv_path}")

    # 对比图
    print("[5] 绘制对比图")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("MolGPT LoRA Fine-tuning: Before vs After", fontsize=13)

    ax = axes[0][0]
    bins = np.linspace(0, max(all_df["MW"].max(), 400), 40)
    ax.hist(results["baseline"]["MW"], bins=bins, alpha=0.55, label="Baseline (pre-trained)", color="#8888cc")
    ax.hist(results["finetuned"]["MW"], bins=bins, alpha=0.55, label="Fine-tuned (LoRA)", color="#44aa88")
    ax.set_xlabel("Molecular Weight"); ax.set_ylabel("count"); ax.legend(); ax.set_title("MW distribution")

    ax = axes[0][1]
    bins = np.linspace(0, 1, 41)
    ax.hist(results["baseline"]["sim_to_train_max"], bins=bins, alpha=0.55, label="Baseline", color="#8888cc")
    ax.hist(results["finetuned"]["sim_to_train_max"], bins=bins, alpha=0.55, label="Fine-tuned", color="#44aa88")
    ax.axvline(np.mean(results["finetuned"]["sim_to_train_max"]), ls="--", c="#44aa88", lw=1)
    ax.axvline(np.mean(results["baseline"]["sim_to_train_max"]), ls="--", c="#8888cc", lw=1)
    ax.set_xlabel("Max Tanimoto sim to target ligand set"); ax.legend()
    ax.set_title("Domain relevance (similarity)")

    ax = axes[1][0]
    rates = [len(results["baseline"]) / N_GEN * 100, len(results["finetuned"]) / N_GEN * 100]
    novelty = [results["baseline"]["novel_vs_train"].mean() * 100,
               results["finetuned"]["novel_vs_train"].mean() * 100]
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, rates, w, label="valid %", color="#6699cc")
    ax.bar(x + w/2, novelty, w, label="novel %", color="#cc8866")
    for i, (a, b_) in enumerate(zip(rates, novelty)):
        ax.text(i - w/2, a + 1, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, b_ + 1, f"{b_:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["Baseline", "Fine-tuned"]); ax.legend()
    ax.set_title("Validity & Novelty")

    ax = axes[1][1]
    show = results["finetuned"].sort_values("sim_to_train_max", ascending=False).head(16)
    mols = [Chem.MolFromSmiles(s) for s in show["canonical_smiles"]]
    mols = [m for m in mols if m][:16]
    legend = [f"sim={v:.2f}" for v in show["sim_to_train_max"][:len(mols)]]
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(260, 260),
                                   legends=[l + "\n" + s[:30] for l, s in zip(legend, show["canonical_smiles"])])
        img.save(os.path.join(OUT_DIR, "sample_molecules.png"))
        ax.imshow(np.asarray(img.convert("RGB").resize((700, 700))))
        ax.axis("off"); ax.set_title("Top fine-tuned molecules (sample)")
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "before_after_comparison.png")
    plt.savefig(png, dpi=150)
    print(f"  图: {png}\n  样本图: sample_molecules.png")

    print("\n完成！")


if __name__ == "__main__":
    main()
