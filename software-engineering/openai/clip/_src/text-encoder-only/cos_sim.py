import numpy as np
from scipy.spatial.distance import cosine

from main import encode_text as encode_text_by_vitl14
from sd import encode_text as encode_text_by_sd_clip


def cos_sim(texts: list[str]):
    text_features_vitl14 = encode_text_by_vitl14(texts, "ViT-L/14")
    text_features_sd_clip = encode_text_by_sd_clip(texts, "model/clip.pt")

    # Normalize the features
    text_features_vitl14 /= text_features_vitl14.norm(dim=-1, keepdim=True)
    text_features_sd_clip /= text_features_sd_clip.norm(dim=-1, keepdim=True)

    # Convert to numpy for easier comparison
    text_features_vitl14 = text_features_vitl14.cpu().numpy()
    text_features_sd_clip = text_features_sd_clip.cpu().numpy()

    # Calculate cosine similarity
    print(f"flatten: {text_features_vitl14.flatten().shape}, {text_features_sd_clip.flatten().shape}")
    distance = cosine(text_features_vitl14.flatten(), text_features_sd_clip.flatten())
    similarity = 1 - distance

    print(f"{similarity=}, {distance=}")


if __name__ == "__main__":
    cos_sim(["a diagram", "a dog", "a cat"])
