# 用于对数据做预处理，用于自动按照line by line处理数据
# 不足max length的位置，加padding补全
# 脚本用途：批量预处理，将目录下的 JSONL 分片转换为 Hugging Face Arrow 格式

#!/usr/bin/env python3
# coding: utf-8
"""
Batch preprocess: tokenize JSONL files in train/valid directories and save to Arrow format.
Usage:

目前本代码只适用于mixtral!!!!!!!!!!!!!!!!!!  flash attention 2 要求左侧padding

# 注意，这里的mask_token_type一定要根据run_mntp.py中加载tokenizer的具体行为确定，尤其目前我们的tokenizer非常不规范
1. 1b moe：/mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/onehot-mix1b-4n-8k-315b-b1-256-tp1pp1ep1-iter_0157014 中，
    mask token就没有成功注册！！special token中没有！！！！所以这里要手动的指定是mask，让其与run_mntp.py中逻辑一致
2. 1b dense：/mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/dense_model/8k-1b-dense-4node-gb32-onehot_convert_0731_fixMistake
    mask token就没有问题，此时设为mask

python preprocess_tokenize.py \
    --input_train_dir /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/qxz_30B_jsonl \
    --input_valid_dir /mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/origin_data/val_o \
    --output_train_dir /mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/tokenized_data/train_ds \
    --output_valid_dir /mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/tokenized_data/val_ds \
    --model_name_or_path /mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/onehot-mix1b-4n-8k-315b-b1-256-tp1pp1ep1-iter_0157014 \
    --max_seq_length 8192 \
    --line_by_line \
    --pad_to_max_length \
    --num_proc 32 \
    --cache_dir /mnt/zzbnew/peixunban/qixianzhi/LLM2VEC/model_300b/tokenized_data/cache \
    --mask_token_type blank \
    --flash_attention_2 
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from loguru import logger

def tokenize_and_save_shards(input_dir: Path, output_dir: Path, tokenizer, args):
    files = sorted(input_dir.glob('*.jsonl'))
    if not files:
        raise ValueError(f"No .jsonl files in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for shard_path in tqdm(files, desc=f"Tokenizing shards in {input_dir.name}"):
        shard_dir = output_dir / shard_path.stem
        if shard_dir.exists() and any(shard_dir.iterdir()):
            tqdm.write(f"Skipping processed shard: {shard_path.name}")
            continue

        load_kwargs = {'data_files': {'data': str(shard_path)}, 'split': 'data'}
        if args.cache_dir:
            load_kwargs['cache_dir'] = args.cache_dir
        ds = load_dataset('json', **load_kwargs)

        # —— 新增 —— 只保留 text 列，删掉其他所有字段
        other_cols = [c for c in ds.column_names if c != 'text']
        if other_cols:
            ds = ds.remove_columns(other_cols)

        if 'text' not in ds.column_names:
            raise ValueError(f"Shard {shard_path.name} missing 'text' column; found: {ds.column_names}")
        col = 'text'

        def tokenize_fn(examples):
            texts = examples[col]
            if args.line_by_line:
                texts = [t for t in texts if t and not t.isspace()]
            return tokenizer(
                texts,
                add_special_tokens=False,
                padding='max_length' if args.pad_to_max_length else False,
                truncation=True,
                max_length=args.max_seq_length,
                return_special_tokens_mask=True,
                return_token_type_ids=False
            )

        tokenized = ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=[col],
            num_proc=args.num_proc,
            load_from_cache_file=False,
            desc=f"Tokenizing {shard_path.name}"
        )
        tokenized.save_to_disk(str(shard_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess JSONL to tokenized Arrow shards.")
    parser.add_argument('--input_train_dir', required=True)
    parser.add_argument('--input_valid_dir', required=True)
    parser.add_argument('--output_train_dir', required=True)
    parser.add_argument('--output_valid_dir', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--max_seq_length', type=int, default=8192)
    parser.add_argument('--line_by_line', action='store_true')
    parser.add_argument('--pad_to_max_length', action='store_true')
    parser.add_argument('--num_proc', type=int, default=4)
    parser.add_argument('--cache_dir', type=str, default=None, help='Optional HF datasets cache_dir')
    
    parser.add_argument(
        '--mask_token_type',
        choices=['blank', 'eos', 'mask'],
        default='blank',
        help="How to set mask_token if missing (blank/_ , eos, or mask/<mask>)"
    )
    
    # — modified: 新增 Flash Attention 2 开关
    parser.add_argument(
        '--flash_attention_2',
        action='store_true',
        help="If set, use left-side padding (required for Flash Attention 2)."
    )
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True
    )
    
    # — modified: 根据 flash_attention_2 参数强制设置 padding_side
    if args.flash_attention_2:
        tokenizer.padding_side = "left"
        logger.info(f"Flash Attention 2 enabled → tokenizer.padding_side set to 'left'")
    else:
        logger.info(f"tokenizer.padding_side remains '{tokenizer.padding_side}'")


    if tokenizer.mask_token is None:
        logger.warning("miss mask token, applying mask_token_type logic")
        if args.mask_token_type == "blank":
            tokenizer.mask_token = "_"
        elif args.mask_token_type == "eos":
            tokenizer.mask_token = tokenizer.eos_token
        elif args.mask_token_type == "mask":
            tokenizer.add_tokens(["<mask>"])
            tokenizer.mask_token = "<mask>"
        else:
            raise ValueError(f"mask_token_type {args.mask_token_type} is not supported.")
        logger.info(f"set mask_token to: {tokenizer.mask_token}")
    else:
        logger.info(f"there is mask token: {tokenizer.mask_token}")

    if tokenizer.pad_token is None:
        logger.warning("miss pad token, using eos_token as pad_token")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"set pad_token to: {tokenizer.pad_token}")
    else:
        logger.info(f"there is pad token: {tokenizer.pad_token}")

    tokenize_and_save_shards(Path(args.input_train_dir), Path(args.output_train_dir), tokenizer, args)
    tokenize_and_save_shards(Path(args.input_valid_dir), Path(args.output_valid_dir), tokenizer, args)
