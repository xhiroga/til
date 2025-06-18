import torch
import torch_tensorrt
import torchvision.models as models
from torch.fx.passes import graph_drawer


# Load TensorRT compiled model
trt_model = torch_tensorrt.load("models/trt.pt2")

# TensorRT models are not FX GraphModules, so we can't visualize them directly
# Instead, let's create a visualization of the original model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Export the model to get FX graph
example_input = torch.randn(1, 3, 224, 224)
ep = torch.export.export(model, (example_input,))
gm = ep.module()

drawer = graph_drawer.FxGraphDrawer(gm, "resnet50_graph")
with open("output/resnet50_graph.svg", "wb") as f:
    f.write(drawer.get_dot_graph().create_svg())
