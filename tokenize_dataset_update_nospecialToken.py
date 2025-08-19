from pathlib import Path
import multiprocessing as mp

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

JSON_DIR = Path(
    "/mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/training_data/raw_data_val"
)
OUT_DIR = (
    "/mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/training_data/"
    "tokenized_files/val"
)

MAX_LEN     = 8192
N_PROC      = 10      # ★ 40 核 / 4 线程 ≈ 10 进程 最平衡
BATCH_TOK   = 4096    # ★ 一次吞 4k 行，大幅压 I/O

# 加载 HF tokenizer（已经含有 <pad> token）
tok = AutoTokenizer.from_pretrained(
    "/mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/dense_model/"
    "8k-1b-dense-4node-gb32-onehot_convert_0731_fixMistake",
    use_fast=True,
)
# 确保 pad_token 正确
tok.pad_token = tok.pad_token or "<pad>"
tok.model_max_length = MAX_LEN

def tok_fn(batch):
    # 逐行 token 化，不加 special tokens，截断或 pad 到 MAX_LEN
    return tok(
        batch["text"],
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

def process_split(json_paths, out_dir):
    shards = []
    for p in sorted(json_paths):
        ds = load_dataset("json", data_files=str(p), split="train")
        ds = ds.map(
            tok_fn,
            batched=True,
            batch_size=BATCH_TOK,
            num_proc=N_PROC,
            remove_columns=["text"],
            desc="Tokenizing lines",
        )
        shards.append(ds)

    # 合并所有 shard 并写出到 disk
    full = concatenate_datasets(shards)
    full.save_to_disk(out_dir, max_shard_size="4GB")
    print(f"✓ Saved tokenized data to {out_dir}")

if __name__ == "__main__":
    all_json = list(JSON_DIR.glob("*.json"))
    process_split(all_json, OUT_DIR)

