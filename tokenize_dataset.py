from pathlib import Path
from itertools import chain
import multiprocessing as mp

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

JSON_DIR       = Path("/mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/training_data/raw_data_val")
OUT_DIR = "/mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/training_data/tokenized_files/val"                # 输出目录

MAX_LEN        = 8192
N_PROC         = 10                       # ★ 40 核 / 4 线程 ≈ 10 进程 最平衡
BATCH_TOK      = 4096                     # ★ 一次吞 4 k 行，大幅压 I/O
BATCH_GROUP    = 8192                     # ★ group_texts 再放大

tok = AutoTokenizer.from_pretrained("/mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/dense_model/8k-1b-dense-4node-gb32-onehot_convert_0730_fixMistake", use_fast=True)

def tok_fn(batch):
    return tok(batch["text"],
               return_special_tokens_mask=True,
               return_token_type_ids=False)

def group_texts(examples):
    concat = {k: list(chain(*examples[k])) for k in examples}
    total = (len(concat["input_ids"]) // MAX_LEN) * MAX_LEN
    if total == 0:
        return {}
    return {
        k: [concat[k][i:i+MAX_LEN] for i in range(0, total, MAX_LEN)]
        for k in concat
    }

def process_split(json_paths, out_dir):
    datasets = []
    for p in json_paths:
        ds = (load_dataset("json", data_files=str(p), split="train")
              .map(tok_fn,   batched=True, batch_size=BATCH_TOK,
                   num_proc=N_PROC, remove_columns=["text"])
              .map(group_texts, batched=True, batch_size=BATCH_GROUP,
                   num_proc=N_PROC, desc="group_texts"))
        datasets.append(ds)

    concatenate_datasets(datasets).save_to_disk(
        out_dir, max_shard_size="4GB")     # ★ 写出时每分片 4 GB，I/O 更流畅
    print(f"✓ Saved {out_dir}")

all_json   = sorted(JSON_DIR.glob("*.json"))
process_split(all_json, OUT_DIR)

