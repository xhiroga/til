import torch
import clip


def encode_text(texts: list[str], model_name: str) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(model_name, device=device)
    model.vision = None

    # Original code refers self.visual.conv1.weight.dtype
    model.__class__.dtype = property(lambda obj: obj.transformer.resblocks[0].attn.in_proj_weight.dtype)

    text = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        print(f"{text_features.shape=}")

    return text_features

if __name__ == "__main__":
    encode_text(["a diagram", "a dog", "a cat"], "ViT-L/14")
