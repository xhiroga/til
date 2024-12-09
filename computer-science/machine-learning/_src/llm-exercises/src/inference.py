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


def main(
    model_name: str,
    test_dataset_names: list[str] = ["elyza/ELYZA-tasks-100", "elyza-tasks-100-TV_0"],
    few_shot_prompting: bool = False,
):
    test_prompt = """\
200字程度で簡潔に回答してください。

# 指示
{}
# 回答
"""
    if few_shot_prompting and (few_shot_prompt := os.environ.get("FEW_SHOT_PROMPT")):
        test_prompt = few_shot_prompt + test_prompt

    config = {
        "base_model_id": model_name,
        "mode": "inference",
        "test": {
            "datasets": test_dataset_names,
            "prompt": test_prompt,
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
        test_dataset_names,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--test_dataset_names", type=str, nargs="+")
    parser.add_argument("--few_shot_prompting", action="store_true")
    args = parser.parse_args()
    main(
        args.model_name,
        args.test_dataset_names,
        args.few_shot_prompting,
    )
