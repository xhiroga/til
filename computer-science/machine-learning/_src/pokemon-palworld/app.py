import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from utils.SimpleCNN import SimpleCNN

image_size = 256

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Load the trained model
model = SimpleCNN(image_size=image_size)
model.load_state_dict(torch.load('models/SimpleCNN_1_256x256.pth'))
model.eval()

def classify_image(input_image: Image):
    # Convert the input image to tensor
    input_tensor = transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    
    # Forward pass the input through the model
    output = model(input_tensor)
    
    # Get the predicted class
    predicted_class = torch.argmax(output, dim=1).item()
    
    # Return the class name
    return 'Pokemon' if predicted_class == 0 else 'Pal'

# Define the Gradio interface
demo = gr.Interface(fn=classify_image, inputs=gr.Image(type='pil'), outputs='text')

# Launch the application
demo.launch()
