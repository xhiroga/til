if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#][^=]*)=(.*)$") {
            $value = $matches[2].Trim('"')
            [System.Environment]::SetEnvironmentVariable($matches[1], $value, "Process")
        }
    }
}

python -c "from musubi_tuner import fpack_generate_video; fpack_generate_video.main()" `
    --dit "$env:MODELS/diffusion_models/FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors" `
    --vae "$env:MODELS/vae/diffusion_pytorch_model.safetensors" `
    --text_encoder1 "$env:MODELS/text_encoder/model-00001-of-00004.safetensors" `
    --text_encoder2 "$env:MODELS/text_encoder_2/model.safetensors" `
    --image_encoder "$env:MODELS/image_encoder/model.safetensors" `
    --image_path $env:IMAGE `
    --prompt "rotating 360 degrees" `
    --video_size 960 544 --fps 30 --infer_steps 25 `
    $env:OPTIONS.Split(' ') `
    --device cuda `
    --attn_mode sdpa --fp8_scaled `
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 `
    --save_path output --output_type both `
    --seed 1234 --lora_multiplier 1.5 --lora_weight $env:LORA
