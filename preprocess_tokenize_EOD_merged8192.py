# # 用于对数据做预处理，首先读取数据，识别基因切换位置
# # 基因末尾自动加<EOD>。采用merge思路，用下一段基因位置，补全8192。换句话说，首先将每大段基因，末尾加入EOD，然后所有数据首尾组成数据流，然后tokenization后，再用8192长度划船切出。
# # 脚本用途：批量预处理，将目录下的 JSONL 分片转换为 Hugging Face Arrow 格式

# #!/usr/bin/env python3
# # coding: utf-8
# """
# Batch preprocess: tokenize JSONL files in train/valid directories and save to Arrow format.
# Usage:

# python preprocess_tokenize_EOD_merged8192.py \
#   --input_dir /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/qxz_30B_jsonl \
#   --output_dir /mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/1b_moe_new/tokenized_data \
#   --model_name_or_path /mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/1b_moe_new/pretrained_model/Mixtral_onehot_mix_1b_16n_8k293B_eod_111_pai_0805 \
#   --max_seq_length 8192 \
#   --num_proc 16

# """

# #!/usr/bin/env python3
# # coding: utf-8
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset
from loguru import logger

def window_token_units_gen(long_str, window_size):
    buf = []
    i = 0
    n = len(long_str)
    while i < n:
        if long_str.startswith("<EOD>", i):
            buf.append("<EOD>")
            i += 5
        else:
            buf.append(long_str[i])
            i += 1
        if len(buf) == window_size:
            yield ''.join(buf)
            buf.clear()
    # 不足window_size的部分丢弃

def make_eod_longstring(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chrom, start, end, seg_idx = obj["src"].split("_")
            records.append((chrom, start, end, int(seg_idx), obj["text"]))
    if not records:
        return ""

    # 补<EOD>
    texts = []
    for i, rec in enumerate(records):
        chrom, start, end, seg_idx, text = rec
        is_last = (i == len(records)-1 or
                   (records[i+1][0], records[i+1][1], records[i+1][2]) != (chrom, start, end))
        if is_last:
            text = text + "<EOD>"
        texts.append(text)
    return "".join(texts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True
    )
    if "<EOD>" not in tokenizer.get_vocab():
        logger.error(f"No EOD! break!!!!!!!!!!!!!!!!!! EROROR")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<EOD>"]})

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        count = count + 1
        logger.info(f"Processing {jsonl_file.name}")
        long_str = make_eod_longstring(jsonl_file)
        if not long_str:
            logger.warning(f"{jsonl_file} 为空，跳过。")
            continue

        # 用生成器滑窗，不拉list
        logger.info("切窗（生成器）")
        windows = list(window_token_units_gen(long_str, args.max_seq_length))
        logger.info(f"窗口数: {len(windows)}")
        if not windows:
            logger.warning(f"{jsonl_file} 生成窗口为空，跳过。")
            continue

        ds = Dataset.from_dict({"text": windows})
        logger.info("开始tokenizer并行map")
        def tokenize_fn(batch):
            out = tokenizer(
                batch["text"],
                add_special_tokens=False,
                padding="do_not_pad",
                truncation=False
            )
            return {"input_ids": out["input_ids"]}
        ds = ds.map(
            tokenize_fn,
            batched=True,
            batch_size=4096,
            num_proc=args.num_proc,
            remove_columns=["text"]
        )
        
        
        if count == 1:
            logger.info("DEBUG:二次filter窗口长度")
            before_len = len(ds)
            ds = ds.filter(lambda ex: len(ex["input_ids"]) == args.max_seq_length)
            after_len = len(ds)

            print(f"过滤前样本数: {before_len}")
            print(f"过滤后样本数: {after_len}")
            print(f"被过滤掉的样本数: {before_len - after_len}")

        out_path = output_dir / jsonl_file.stem
        ds.save_to_disk(str(out_path))
        logger.info(f"→ 已保存到 {out_path}")

if __name__ == "__main__":
    main()

