# https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
import torch
import torchvision.models as models
from torch.profiler import ProfilerActivity, profile

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(
    activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
) as prof:
    model(inputs)

# prof オブジェクトに with ブロック外でアクセスしているが、これが正解。
# ブロック内でアクセスすると何も表示されない。
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
