# Packaging musubi-tuner

## Install

```sh
touch .env
uv init --python 3.10
uv add git+https://github.com/xhiroga/musubi-tuner-xhiroga
export $(cat .env | xargs) && uv run main.py \
    --dit ${MODELS}/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
    --vae ${MODELS}/vae/diffusion_pytorch_model.safetensors \
    --text_encoder1 ${MODELS}/text_encoder/model-00001-of-00004.safetensors \
    --text_encoder2 ${MODELS}/text_encoder_2/model.safetensors \
    --image_encoder ${MODELS}/image_encoder/model.safetensors \
    --image_path ${IMAGE} \
    --prompt "rotating 360 degrees" \
    --video_size 960 544 --video_seconds 3 --fps 30 --infer_steps 25 \
    --device cuda \
    --attn_mode sdpa --fp8_scaled \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
    --save_path output --output_type both \
    --seed 1234 --lora_multiplier 1.5 --lora_weight ${LORA}
```
