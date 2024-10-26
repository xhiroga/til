import os
import clip
import torch
from dotenv import load_dotenv
from safetensors import safe_open


load_dotenv()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = "ViT-L/14"
    print(f"{clip_model=}")
    model, _ = clip.load(clip_model, device=device)
    for k, v in model.state_dict().items():
        print(f"{k=}, {v.shape=}")

    print("model.state_dict:")
    state_dict = model.state_dict()
    for k in state_dict.keys():
        print(f"{k=}, {state_dict[k].shape=}")

    sd_model_path = os.environ.get("SD_MODEL_PATH")
    print(f"{sd_model_path=}")
    with safe_open(sd_model_path, framework="pt", device=0) as f:
        for k in f.keys():
            print(f"{k=}, {f.get_tensor(k).shape=}")


if __name__ == "__main__":
    main()
