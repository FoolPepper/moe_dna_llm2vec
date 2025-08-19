#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The script is adapted from https://github.com/huggingface/transformers/blob/51bcadc10a569847b93a30dbe3a077037ae63bad/examples/pytorch/language-modeling/run_mlm.py
"""

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Any, Tuple, List
import numpy as np

import datasets
import evaluate
from datasets import load_dataset

import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, get_peft_model

from llm2vec.models import (
    MistralBiForMNTP,
    LlamaBiForMNTP,
    GemmaBiForMNTP,
    Qwen2BiForMNTP,
    MixtralBiForMNTP,
)

# 让代码直接读取tokenization后的数据
from datasets import load_from_disk, Dataset, concatenate_datasets
import inspect           # 仅用于跳过后面代码的小技巧

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.38.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_model_class(config):
    config_class_name = config.__class__.__name__
    if config_class_name == "MistralConfig":
        return MistralBiForMNTP
    elif config_class_name == "LlamaConfig":
        return LlamaBiForMNTP
    elif config_class_name == "GemmaConfig":
        return GemmaBiForMNTP
    elif config_class_name == "Qwen2Config":
        return Qwen2BiForMNTP
    elif config_class_name == "MixtralConfig":
        return MixtralBiForMNTP
    else:
        raise ValueError(f"Model class {config_class_name} not supported.")


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    name = model.config.__class__.__name__
    if lora_modules is None and name in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
        "MixtralConfig"
    ]:
        if name == "MixtralConfig":
            # 先只打注意力层，最稳，显存开销小、不碰 MoE 专家层
            lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            # 可选：如果要覆盖MoE专家前馈
            # lora_modules = ["q_proj","k_proj","v_proj","o_proj","w1","w2","w3","gate"]
        else:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    logger.info(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )

        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            # 判断一个目录是不是arrow数据集, 判断标准为存在dataset_info.json
            # def _is_arrow_dir(path: str) -> bool:
            #     return os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json"))
            
            # 如果 train_file 是目录，则检查它的子目录
            if self.train_file and os.path.isdir(self.train_file):
                child_dirs = [
                    os.path.join(self.train_file, d)
                    for d in sorted(os.listdir(self.train_file))
                    if os.path.isdir(os.path.join(self.train_file, d))
                    and os.path.exists(os.path.join(self.train_file, d, "dataset_info.json"))
                ]
                if not child_dirs:
                    raise ValueError(
                        f"`train_file` 指定的目录 `{self.train_file}` 下必须包含一个或多个子目录，"
                        "每个子目录都应是 HuggingFace Arrow 数据集（包含 dataset_info.json）。"
                    )
                else:
                    logger.info("qi debug: the post check has confirmed the train file is a HF arrow dirs")
                # 如果指定了 validation_file，也做同样的检查
                if self.validation_file and os.path.isdir(self.validation_file):
                    val_child_dirs = [
                        os.path.join(self.validation_file, d)
                        for d in sorted(os.listdir(self.validation_file))
                        if os.path.isdir(os.path.join(self.validation_file, d))
                        and os.path.exists(os.path.join(self.validation_file, d, "dataset_info.json"))
                    ]
                    if not val_child_dirs:
                        raise ValueError(
                            f"`validation_file` 指定的目录 `{self.validation_file}` 下必须包含一个或多个子目录，"
                            "每个子目录都应是 HuggingFace Arrow 数据集（包含 dataset_info.json）。"
                        )
                # 校验完毕，直接返回，不再做后续后缀检查
                print("Detected directory-of-arrow-subdirs for train/valid → 跳过后续扩展名检查")
                return
            
            logger.info("qi debug: into the text files legal check!")
            # 官方情况：对应train file是文件的情况       
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`train_file` should be a csv, a json or a txt file."
                    )
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`validation_file` should be a csv, a json or a txt file."
                    )


# add more arguments
@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    mask_token_type: str = field(
        default="blank",
        metadata={"help": "The type of mask token. Options: blank, eos, mask"},
    )

    # stop_after_n_steps: int = field(
    #     default=10000, metadata={"help": "Stop training after n steps"}
    # )
    
    # # qi debug: 将stop变成可选，如不显示设置不影响训练 
    stop_after_n_steps: Optional[int] = field(
        default=None, metadata={"help": "Stop training after n steps (None to disable)"}
    )

    data_collator_type: str = field(
        default="default",
        metadata={"help": "The type of data collator. Options: default, all_mask"},
    )


class DataCollatorForLanguageModelingWithFullMasking(DataCollatorForLanguageModeling):
    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 100% MASK, 0% random, 0% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class MNTPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["labels"]

    def _remove_unused_columns(
        self, dataset: "datasets.Dataset", description: Optional[str] = None
    ):
        return dataset

    # We need a custom save function as we have to save the inner model
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # model organization is MODEL_TYPEBiForMNTP.model -> MODEL_TYPELBiModel, we have to save the inner model, handled by save_peft_model function of the outer model
        self.model.save_peft_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()

    add_special = False       # qi debug, 硬编码不用分隔符
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
    else:
        logger.info("Get into the user dataload")
        def maybe_load_pretok(path: str, split_name: str):
            """
            支持一种情况：
            1) path为目录，且path下的子目录都是HF的放置有arrow的目录，如果设置了 streaming=True，则对父目录下所有子目录中的 .arrow 文件做流式加载
            -- 父目录
            |
            -----子目录
            |
            ------- .arrow
            ------- dataset_info.json
            - 否则返回 None.
            """
            if path is None:
                return None
        
            # 2) 目录下含多个 Arrow 数据集子目录
            if os.path.isdir(path):
                # 合法性判别
                child_dirs = [
                    os.path.join(path, d)
                    for d in sorted(os.listdir(path))
                    if os.path.isdir(os.path.join(path, d))
                    and os.path.exists(os.path.join(path, d, "dataset_info.json"))
                ]
                
                if child_dirs and not data_args.streaming:
                    ds_list = [load_from_disk(d) for d in child_dirs]
                    return concatenate_datasets(ds_list, axis=0)

                if child_dirs and data_args.streaming:
                    # 流式读取所有子目录中的 .arrow 文件
                    ds_dict = load_dataset(
                        "arrow",
                        data_files={split_name: os.path.join(path, "**", "*.arrow")},
                        streaming=True,
                    )
                    return ds_dict[split_name]

            # 其余情况交给默认逻辑
            return None


        # 判断是否 
        raw_train = maybe_load_pretok(data_args.train_file, "train")
        raw_valid = maybe_load_pretok(data_args.validation_file, "validation")
        
        if raw_train is not None: 
            logger.info("Qi debug: Arrow dataset has been prepared!! Data has been tokenization offload!")  
            raw_datasets = {"train": raw_train}
            if raw_valid is not None:
                raw_datasets["validation"] = raw_valid           
        else:
            # llm2vec原逻辑
            logger.info("Qi debug: no array dataset, train start with the origin text/csv/json files")
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
                extension = data_args.train_file.split(".")[-1]
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
                extension = data_args.validation_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # qi debug：如果启动了flash attention 2 则pad在左侧
    if model_args.attn_implementation == "flash_attention_2":
        logger.warning("start flash attention 2, padding side set to left!!!")
        tokenizer.padding_side = "left"
    
    # blank, eos, mask
    if tokenizer.mask_token is None:
        logger.error("qi debug: mask token is None!")
        if custom_args.mask_token_type == "blank":
            logger.error("qi debug: use blank as mask")
            tokenizer.mask_token = "_"
        elif custom_args.mask_token_type == "eos":
            tokenizer.mask_token = tokenizer.eos_token
        elif custom_args.mask_token_type == "mask":
            tokenizer.add_tokens(["<mask>"])
            tokenizer.mask_token = "<mask>"
        else:
            raise ValueError(
                f"mask_token_type {custom_args.mask_token_type} is not supported."
            )
    else:
        logger.info(f"qi debug: there is mask token: {tokenizer.mask_token}")

    if tokenizer.pad_token is None:
        logger.error("qi debug: pad token is None!")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        logger.info(f"qi debug: there is pad token: {tokenizer.pad_token}")

    # Loading bidirectional model using LLM2Vec package
    model_class = get_model_class(config)
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )

    # model organization is MODEL_TYPEBiForMNTP.model -> MODEL_TYPELBiModel, we have to apply PEFT to the inner model
    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        raise ValueError("qi debug: what the embedding do????????????????????????")
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # 直接读取tokenization逻辑
    # ---------- NEW (B): 如果已有 input_ids，直接跳过全部 tokenize / group_texts ----------
    if "input_ids" in raw_datasets["train"].column_names:
        tokenized_datasets = raw_datasets     # 数据已经是 token 格式
    else:
        if data_args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            logger.info("Qi Note: we have into the line by line !!!! ")
            
            padding = "max_length" if data_args.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples[text_column_name] = [
                    line
                    for line in examples[text_column_name]
                    if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    add_special_tokens=add_special, # qi debug: 硬编码，是否插入</s> </s>, 对于人类基因组训练，调成false
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                    return_token_type_ids=False,
                )

            with training_args.main_process_first(desc="dataset map tokenization"):
                if not data_args.streaming:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=[text_column_name],
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on dataset line_by_line",
                    )
                else:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        remove_columns=[text_column_name],
                    )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(
                    examples[text_column_name], return_special_tokens_mask=True,
                    return_token_type_ids=False 
                )

            with training_args.main_process_first(desc="dataset map tokenization"):
                if not data_args.streaming:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on every text in dataset",
                    )
                else:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        remove_columns=column_names,
                    )

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: list(chain(*examples[k])) for k in examples.keys()
                }
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
                # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
                total_length = (total_length // max_seq_length) * max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [
                        t[i : i + max_seq_length]
                        for i in range(0, total_length, max_seq_length)
                    ]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/process#map

            with training_args.main_process_first(desc="grouping texts together"):
                if not data_args.streaming:
                    tokenized_datasets = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc=f"Grouping texts in chunks of {max_seq_length}",
                    )
                else:
                    tokenized_datasets = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                    )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        # def preprocess_logits_for_metrics(logits, labels):
        #     if isinstance(logits, tuple):
        #         # Depending on the model and config, logits may contain extra tensors,
        #         # like past_key_values, but logits always come first
        #         logits = logits[0]
        #     return logits.argmax(dim=-1)

        # metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        # def compute_metrics(eval_preds):
        #     preds, labels = eval_preds
        #     preds = preds[:, :-1]
        #     labels = labels[:, 1:]
        #     # preds have the same shape as the labels, after the argmax(-1) has been calculated
        #     # by preprocess_logits_for_metrics
        #     labels = labels.reshape(-1)
        #     preds = preds.reshape(-1)
        #     mask = labels != -100
        #     labels = labels[mask]
        #     preds = preds[mask]
        #     return metric.compute(predictions=preds, references=labels)

        # qi debug(支持moe)
        def preprocess_logits_for_metrics(logits, labels):
            # 有的模型会返回 (logits, router_logits, aux_loss) 或 (logits, past_key_values)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            # [B, S, V] -> [B, S]，只保留 token 预测，避免把巨大的 logits 搬到 CPU
            return torch.argmax(logits, dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds  # numpy arrays

            # 兜底：如果 preds 仍是 1D（比如只缓存了 loss），直接返回一个标志，避免崩溃
            if isinstance(preds, np.ndarray) and preds.ndim == 1:
                return {"accuracy": float("nan"), "eval_pred_available": 0.0}

            # 对齐：下一个 token 预测
            preds = preds[:, :-1]
            labels = labels[:, 1:]

            # 展平并按 -100 掩码
            mask = labels != -100
            if mask.sum() == 0:
                return {"accuracy": float("nan"), "masked_tokens": 0.0}

            accuracy = ((preds == labels) & mask).sum() / mask.sum()
            return {
                "accuracy": float(accuracy),
                "masked_tokens": float(mask.sum()),
            }
        
        
    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = (
        data_args.line_by_line
        and training_args.fp16
        and not data_args.pad_to_max_length
    )
    data_collator_cls = None
    if custom_args.data_collator_type == "all_mask":
        data_collator_cls = DataCollatorForLanguageModelingWithFullMasking
    elif custom_args.data_collator_type == "default":
        data_collator_cls = DataCollatorForLanguageModeling
    else:
        raise ValueError(
            f"data_collator_type {custom_args.data_collator_type} is not supported."
        )

    data_collator = data_collator_cls(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # Initialize our Trainer
    trainer = MNTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None
        ),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None
        ),
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
