import gradio as gr
import json
import torch
from PIL import Image
from safetensors import safe_open
from torchvision import models, transforms

with open('data/pokemon-1st-gen-labels.json') as f:
    labels = json.load(f)

model = models.resnet18(pretrained=True)

model.fc = torch.nn.Linear(model.fc.in_features, len(labels))

model_save_path = "models/model.safetensors"
tensors = {}
with safe_open(model_save_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

model.load_state_dict(tensors, strict=False)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_image(input_image: Image):
    img_t = preprocess(input_image)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        output = model(batch_t)
    
    probabilities = torch.nn.functional.softmax(output, dim=1)

    actual_class_size = len(labels) # 143
    label_to_prob = {labels[i]: prob for i, prob in enumerate(probabilities[0])}
    return label_to_prob

demo = gr.Interface(fn=classify_image, inputs=gr.Image(type='pil'), outputs='label')
demo.launch()
