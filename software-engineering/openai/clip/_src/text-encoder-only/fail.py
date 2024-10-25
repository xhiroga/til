import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.visual = None

text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    print(f"{text_features.shape}")

# $ uv run python main.py
# Traceback (most recent call last):
#   File "/home/hiroga/Documents/GitHub/til/software-engineering/openai/clip/_src/text-encoder-only/main.py", line 11, in <module>
#     text_features = model.encode_text(text)
#                     ^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/hiroga/Documents/GitHub/til/software-engineering/openai/clip/_src/text-encoder-only/.venv/lib/python3.12/site-packages/clip/model.py", line 344, in encode_text
#     x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
#                                         ^^^^^^^^^^
#   File "/home/hiroga/Documents/GitHub/til/software-engineering/openai/clip/_src/text-encoder-only/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1729, in __getattr__
#     raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
# AttributeError: 'CLIP' object has no attribute 'dtype'. Did you mean: 'type'?
