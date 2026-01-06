#!/bin/bash

export TINKER_API_KEY=tml-fDEkFax6ugfBorfdw9YYcRq7SXo2uSd4Wp3EKxN3mIbuNHN4rvAzA3FWMeryGUV3PAAAA
python -m tinker_cookbook.recipes.math_rl.train env=polaris group_size=8 groups_per_batch=128 lora_rank=64 learning_rate=3e-5 max_tokens=8192 log_path=base/polaris/Qwen3-4B_polaris_3e-5 wandb_project=RSA wandb_name=Qwen3-4B_polaris_3e-5