# export CUDA_VISIBLE_DEVICES=0,1

export WANDB_PROJECT=llm2vec_mntp_MOE_EOD
export WANDB_NAME=1b_moe_nanhu_8k_30b_withEOD_default_020_lr5e-5_gradientCut05
export WANDB_API_KEY=329524f65708d95f93a052cca09e3d76921132e1

torchrun --nproc_per_node=8 \
  /workspace/llm2vec/experiments/run_mntp.py \
  /workspace/llm2vec/train_configs/mntp/1b_moe_nanhu_8k_30b_withEOD_default_020_lr5e-5_gradientCut05.json