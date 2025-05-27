# Sandbox

LoRAの`network_dim`, `network_alpha` は `.safetensors` ファイルに保存されているのか？

## SDXL LoRA

```console
$ uv run inside_lora.py

[k for k, v in items]=['lora_unet_input_blocks_4_1_proj_in.alpha', 'lora_unet_input_blocks_4_1_proj_in.lora_down.weight', 'lora_unet_input_blocks_4_1_proj_in.lora_up.weight', 'lora_unet_input_blocks_4_1_proj_out.alpha', 'lora_unet_input_blocks_4_1_proj_out.lora_down.weight', 'lora_unet_input_blocks_4_1_proj_out.lora_up.weight', ...]
network_alpha: 32.0, network_dim: 32
```

## FramePack LoRA

```console
$ uv run inside_lora.py

[k for k, v in items]=['lora_unet_single_transformer_blocks_0_attn_to_k.alpha', 'lora_unet_single_transformer_blocks_0_attn_to_k.lora_down.weight', 'lora_unet_single_transformer_blocks_0_attn_to_k.lora_up.weight', 'lora_unet_single_transformer_blocks_0_attn_to_q.alpha', 'lora_unet_single_transformer_blocks_0_attn_to_q.lora_down.weight', 'lora_unet_single_transformer_blocks_0_attn_to_q.lora_up.weight', ...]
network_alpha: 4.0, network_dim: 8
```
