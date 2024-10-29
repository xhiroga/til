import os

import clip
import torch
from clip.model import CLIP
from dotenv import load_dotenv
from safetensors import safe_open

load_dotenv()

def export_clip(sd_model_path: str, base_model_name: str, exported_clip_path: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(base_model_name, device=device)
    state_dict =  model.state_dict()

    with safe_open(sd_model_path, framework="pt", device=0) as f:
        resblocks = {}
        resblocks['positional_embedding'] = f.get_tensor('cond_stage_model.transformer.text_model.embeddings.position_embedding.weight')
        resblocks['text_projection'] = state_dict['text_projection']
        resblocks['logit_scale'] = state_dict['logit_scale']
        resblocks['visual.class_embedding'] = torch.zeros(1024) # TODO
        resblocks['visual.positional_embedding'] = torch.zeros(257, 1024) # TODO
        resblocks['visual.proj'] = torch.zeros(1024, 768) # TODO
        resblocks['visual.conv1.weight'] = torch.zeros(1024, 3, 14, 14) # TODO
        resblocks['visual.ln_pre.weight'] = torch.zeros(1024) # TODO
        resblocks['visual.ln_pre.bias'] = torch.zeros(1024) # TODO
        resblocks['visual.ln_post.weight'] = torch.zeros(1024) # TODO
        resblocks['visual.ln_post.bias'] = torch.zeros(1024)
        resblocks['token_embedding.weight'] = f.get_tensor('cond_stage_model.transformer.text_model.embeddings.token_embedding.weight')
        resblocks['ln_final.weight'] = state_dict['ln_final.weight']
        resblocks['ln_final.bias'] = state_dict['ln_final.bias']

        for i in range(12):
            # Self-attention
            q_weight = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.q_proj.weight')
            k_weight = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.k_proj.weight')
            v_weight = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.v_proj.weight')
            
            q_bias = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.q_proj.bias')
            k_bias = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.k_proj.bias')
            v_bias = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.v_proj.bias')

            # Concatenate Q, K, V weights and biases
            in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

            resblocks[f'transformer.resblocks.{i}.attn.in_proj_weight'] = in_proj_weight
            resblocks[f'transformer.resblocks.{i}.attn.in_proj_bias'] = in_proj_bias

            resblocks[f'transformer.resblocks.{i}.attn.out_proj.weight'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.out_proj.weight')
            resblocks[f'transformer.resblocks.{i}.attn.out_proj.bias'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.out_proj.bias')

            # Layer norms
            resblocks[f'transformer.resblocks.{i}.ln_1.weight'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm1.weight')
            resblocks[f'transformer.resblocks.{i}.ln_1.bias'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm1.bias')
            resblocks[f'transformer.resblocks.{i}.ln_2.weight'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm2.weight')
            resblocks[f'transformer.resblocks.{i}.ln_2.bias'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm2.bias')

            # MLPs
            resblocks[f'transformer.resblocks.{i}.mlp.c_fc.weight'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc1.weight')
            resblocks[f'transformer.resblocks.{i}.mlp.c_fc.bias'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc1.bias')
            resblocks[f'transformer.resblocks.{i}.mlp.c_proj.weight'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc2.weight')
            resblocks[f'transformer.resblocks.{i}.mlp.c_proj.bias'] = f.get_tensor(f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc2.bias')

            # Dummy tensor


        torch.save(resblocks, exported_clip_path)


def encode_text(texts: list[str], exported_clip_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = torch.load(exported_clip_path, map_location=device)

    # Create a new CLIP model instance
    model = CLIP(
        embed_dim=768,  # TODO: 現状の値は state_dict['text_projection'].shape[1] と同じ
        image_resolution=224,
        vision_layers=0,
        vision_width=1024,
        vision_patch_size=14,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12
    )
    
    # Load the state dict into the model
    model.load_state_dict(loaded)
    model.to(device)
    model.eval()
    
    model.visual = None
    model.__class__.dtype = property(lambda obj: obj.transformer.resblocks[0].attn.in_proj_weight.dtype)

    text = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        print(f"{text_features.shape=}")
    
    return text_features


if __name__ == "__main__":
    base_model_name = 'ViT-L/14'
    exported_model_path = "model/clip.pt"
    export_clip(os.environ.get("SD_MODEL_PATH"), base_model_name, exported_model_path)
    encode_text(["a diagram", "a dog", "a cat"], exported_model_path)
