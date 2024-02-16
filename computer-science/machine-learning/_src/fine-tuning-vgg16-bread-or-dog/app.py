import gradio as gr
import torch
from PIL import Image
from safetensors import safe_open
from torchvision import models, transforms

labels = ["bread", "dog"]

model = models.vgg16(pretrained=True)

model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)

model_save_path = "models/model.safetensors"
tensors = {}
with safe_open(model_save_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

model.load_state_dict(tensors, strict=False)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

def classify_image(input_image: Image):
    img_t = preprocess(input_image)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        output = model(batch_t)
    
    probabilities = torch.nn.functional.softmax(output, dim=1)
    label_to_prob = {labels[i]: prob for i, prob in enumerate(probabilities[0])}
    return label_to_prob

demo = gr.Interface(fn=classify_image, inputs=gr.Image(type='pil'), outputs='label')
demo.launch()
