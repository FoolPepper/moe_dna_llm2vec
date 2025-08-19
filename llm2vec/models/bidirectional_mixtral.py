import torch

from transformers import (
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralForCausalLM,
    MixtralConfig,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRMSNorm,
    MixtralAttention,
    MixtralFlashAttention2,
    MixtralSdpaAttention,
    MixtralSparseMoeBlock,
    load_balancing_loss_func,
)
from torch import nn
from transformers.utils import logging
from transformers.cache_utils import Cache, StaticCache, SlidingWindowCache

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from .utils import is_transformers_attn_greater_or_equal_4_43_1

from peft import PeftModel

logger = logging.get_logger(__name__)


class ModifiedMixtralAttention(MixtralAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedMixtralFlashAttention2(MixtralFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedMixtralSdpaAttention(MixtralSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


MIXTRAL_ATTENTION_CLASSES = {
    "eager": ModifiedMixtralAttention,
    "flash_attention_2": ModifiedMixtralFlashAttention2,
    "sdpa": ModifiedMixtralSdpaAttention,
}

# class ModifiedMixtralDecoderLayer(MixtralDecoderLayer):
#     def __init__(self, config: MixtralConfig, layer_idx: int):
#         nn.Module.__init__(self)
#         self.hidden_size = config.hidden_size

#         self.self_attn = MIXTRAL_ATTENTION_CLASSES[config._attn_implementation](
#             config, layer_idx
#         )

#         self.block_sparse_moe = MixtralSparseMoeBlock(config)
#         self.input_layernorm = MixtralRMSNorm(
#             config.hidden_size, eps=config.rms_norm_eps
#         )
#         self.post_attention_layernorm = MixtralRMSNorm(
#             config.hidden_size, eps=config.rms_norm_eps
#         )
class ModifiedMixtralDecoderLayer(MixtralDecoderLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        # 先由父类构建标准结构：包含 block_sparse_moe / layernorm / self_attn 等
        super().__init__(config, layer_idx)
        # 仅把注意力模块替换为“非因果”实现
        self.self_attn = MIXTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        # 我们的 Modified*Attention 在 __init__ 里已经设置了 self.is_causal = False


class MixtralBiModel(MixtralModel):
    _no_split_modules = ["ModifiedMixtralDecoderLayer"]

    def __init__(self, config: MixtralConfig):
        if not is_transformers_attn_greater_or_equal_4_43_1():
            raise ValueError(
                "The current implementation of LlamaEncoderModel follows modeling_llama.py of transformers version >= 4.43.1"
            )
        MixtralPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedMixtralDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from forward() in transformers.models.mixtral.modeling_mixtral.MixtralModel
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # 避免HF签名不一致
        use_cache = past_key_values is not None
        
        if self._attn_implementation == "flash_attention_2":
            if attention_mask is not None and use_cache:
                is_padding_right = (
                    attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                )
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.

        # cache_position must be valid here no matter which cache we use
        past_seen_tokens = cache_position[0] if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # if (
        #     self.config._attn_implementation == "sdpa"
        #     and not (using_static_cache or using_sliding_window_cache)
        #     and not output_attentions
        # ):
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         sliding_window=self.config.sliding_window,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache
        if using_sliding_window_cache:
            target_length = max(sequence_length, self.config.sliding_window)
        # StaticCache
        elif using_static_cache:
            target_length = past_key_values.get_max_length()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.zeros(
                (sequence_length, target_length), dtype=dtype, device=device
            )  # causal_mask = torch.full(
            # (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            # )
            exclude_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            if self.config.sliding_window is not None:
                if (
                    not using_sliding_window_cache
                    or sequence_length > self.config.sliding_window
                ):
                    exclude_mask.bitwise_or_(
                        torch.arange(target_length, device=device)
                        <= (cache_position.reshape(-1, 1) - self.config.sliding_window)
                    )
            causal_mask *= exclude_mask
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.dim() == 2:
                    mask_length = attention_mask.shape[-1]
                    padding_mask = (
                        causal_mask[:, :, :, :mask_length]
                        + attention_mask[:, None, None, :]
                    )
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[
                        :, :, :, :mask_length
                    ].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


class MixtralBiForMNTP(MixtralForCausalLM):
    def __init__(self, config):
        MixtralPreTrainedModel.__init__(self, config)
        self.model = MixtralBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # >>> 必须补的属性，HF forward 里会读取 <<<
        self.num_experts = getattr(config, "num_local_experts", None)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", None)
        # 保险起见，再加一个同义名（有些分支用 top_k）
        self.top_k = self.num_experts_per_tok
        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.0)
        self.output_router_logits = getattr(config, "output_router_logits", False)
        self.router_jitter_noise = getattr(config, "router_jitter_noise", 0.0)
             
        # Initialize weights and apply final processing
        self.post_init()
    
    
    def forward(self, *args, output_router_logits=None, **kwargs):
        # 统一策略：训练开路由，评测关路由
        if output_router_logits is None:
            output_router_logits = self.output_router_logits
        if not self.training:
            output_router_logits = False
        kwargs["output_router_logits"] = output_router_logits
        return super().forward(*args, **kwargs)


    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)