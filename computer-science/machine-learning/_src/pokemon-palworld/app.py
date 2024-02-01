import gradio as gr
import torch
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from utils.SimpleCNN import SimpleCNN

image_size = 256

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Load the trained model
model_save_path = "models/model.safetensors"
tensors = {}
with safe_open(model_save_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

model = SimpleCNN(image_size=image_size)
model.load_state_dict(tensors)

model.eval()

def classify_image(input_image: Image):
    # Convert the input image to tensor
    input_tensor = transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    
    # Forward pass the input through the model
    output = model(input_tensor)
    print(output)
    
    probabilities = torch.nn.functional.softmax(output, dim=1)
    
    return {'Pokemon': probabilities[0][0], 'Pal': probabilities[0][1]}

# Define the Gradio interface
demo = gr.Interface(fn=classify_image, inputs=gr.Image(type='pil'), outputs='label')

# Launch the application
demo.launch()
