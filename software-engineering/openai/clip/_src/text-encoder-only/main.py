import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.__class__.dtype = property(lambda obj: obj.transformer.resblocks[0].attn.in_proj_weight.dtype)

text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    print(f"{text_features.shape}")
