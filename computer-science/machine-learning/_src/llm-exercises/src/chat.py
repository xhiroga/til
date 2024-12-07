import logging
import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import DEFAULT_TEST_PROMPT

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def find_safetensor_dirs(base_dir):
    safetensor_dirs = []
    for root, _dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".safetensors"):
                relative_path = os.path.relpath(root, base_dir)
                safetensor_dirs.append(relative_path)
                break
    return safetensor_dirs


# Define the available models by listing subdirectories in the models directory
models_dir = "models"
models = sorted(
    find_safetensor_dirs(models_dir),
    key=lambda d: os.path.getctime(os.path.join(models_dir, d)),
    reverse=True,
)

# Load the default model and tokenizer
model_name = models[0]


def model_loader():
    current_model_name = None
    model = None
    tokenizer = None

    while True:
        model_name = yield model, tokenizer
        logger.debug(f"Received {model_name=}")
        if model_name == current_model_name:
            logger.debug(f"Already loaded {model_name=}")
            continue
        else:
            logger.debug(f"Loading {model_name=}")
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(models_dir, model_name)
            )
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(models_dir, model_name)
            )
            current_model_name = model_name
            logger.debug(f"Loaded {model_name=}")


loader = model_loader()
next(loader)


def respond(
    message,
    _history: list[tuple[str, str]],
    context,
    model_half: bool = False,
    max_new_tokens,
    temperature,
    model_name,
):
    global loader
    logger.debug(f"Respond function called with {model_name=}")
    model, tokenizer = loader.send(model_name)
    logger.debug(f"Model and tokenizer loaded: {model_name=}")

    instruction = context + "\n" + DEFAULT_TEST_PROMPT.format(message)
    logger.debug(f"{instruction=}")

    if model_half:
        model.half()

    tokenized_input = tokenizer.encode(
        instruction, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    attention_mask = torch.ones_like(tokenized_input)
    logger.debug(f"{tokenized_input=}")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokenized_input,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
        )
        response = tokenizer.decode(
            outputs[tokenized_input.size(1) :], skip_special_tokens=True
        )
        logger.debug(f"{response=}")

    return response


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="", label="Context"),
        gr.Checkbox(value=False, label="Half precision"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Dropdown(choices=models, value=models[0], label="Model name"),
    ],
)

if __name__ == "__main__":
    demo.launch()
