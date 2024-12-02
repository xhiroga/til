import argparse
import os

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import infer

assert os.environ.get("WANDB_ENTITY")
assert os.environ.get("WANDB_PROJECT")
model_dir = "models"

load_dotenv()


def main(model_name: str):
    config = {
        "base_model_id": model_name,
        "test": {
            "datasets": ["elyza/ELYZA-tasks-100", "elyza-tasks-100-TV_0"],
            "prompt": """\
# 指示
200字程度で簡潔に回答してください。
{}
# 回答
""",
            "limit": 5,
            "model_half": True,
            "max_new_tokens": 200,
        },
    }

    wandb.init(config=config)
    run_name = wandb.run.name

    model = AutoModelForCausalLM.from_pretrained(
        f"{model_dir}/{model_name}", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    )
    inference = infer(
        model,
        tokenizer,
        ["elyza/ELYZA-tasks-100", "elyza-tasks-100-TV_0"],
        model_name,
        run_name=run_name,
        test_prompt=config["test"]["prompt"],
        test_limit=config["test"]["limit"],
        model_half=config["test"]["model_half"],
        max_new_tokens=config["test"]["max_new_tokens"],
    )
    wandb.log(inference)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a model for inference.")
    parser.add_argument(
        "--model_name", type=str, help="The name of the model to use for inference."
    )
    args = parser.parse_args()
    main(args.model_name)
