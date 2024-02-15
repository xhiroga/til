import gradio as gr
import json
import torch
from PIL import Image
from torchvision import models, transforms

with open("data/imagenet-simple-labels.json") as f:
    labels = json.load(f)

model = models.vgg16(pretrained=True)
model.eval()  # 推論モードに設定

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def classify_image(input_image: Image):
    img_t = preprocess(input_image)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        output = model(batch_t)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    label_to_prob = {labels[i]: prob for i, prob in enumerate(probabilities[0])}
    return label_to_prob


demo = gr.Interface(fn=classify_image, inputs=gr.Image(type="pil"), outputs="label")
demo.launch()
