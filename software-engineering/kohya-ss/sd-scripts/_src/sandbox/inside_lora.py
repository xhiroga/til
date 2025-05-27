import os
from dotenv import load_dotenv
from safetensors.torch import load_file

load_dotenv()

lora = load_file(os.getenv('LORA_PATH'))
items = list(lora.items())
print(f"{[k for k, v in items]=}")

alpha = next((v for k, v in items if 'alpha' in k), None)
dim = next((v.size()[0] for k, v in items if 'lora_down' in k), None)
print(f'network_alpha: {alpha}, network_dim: {dim}')
