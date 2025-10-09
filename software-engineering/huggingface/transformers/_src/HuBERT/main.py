from transformers import HubertModel

model = HubertModel.from_pretrained("facebook/hubert-base-ls960", attn_implementation="eager")

print(model)                      # モジュール構成
print(model.config)               # コンフィグ
# パラメータ名と形状
for n, p in model.named_parameters():
    print(n, tuple(p.shape))
