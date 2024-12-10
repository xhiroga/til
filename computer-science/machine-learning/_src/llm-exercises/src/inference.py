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
    test_dataset_names: list[str],
    test_limit: int,
    model_half: bool,
    few_shot_prompting: bool,
):
    test_prompt = """\
### 指示
200字程度で簡潔に回答してください。
{}
### 回答
"""
    if few_shot_prompting and (few_shot_prompt := os.environ.get("FEW_SHOT_PROMPT")):
        test_prompt = few_shot_prompt + test_prompt

    config = {
        "base_model_id": model_name,
        "mode": "inference",
        "test": {
            "datasets": test_dataset_names,
            "prompt": test_prompt,
            "limit": test_limit,
            "model_half": model_half,
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
        config["test"]["datasets"],
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
    parser.add_argument("--model_name", type=str, default="llm-jp/llm-jp-3-1.8b")
    parser.add_argument("--test_dataset_names", type=str, nargs="+", default=["elyza/ELYZA-tasks-100", "elyza-tasks-100-TV_0"])
    parser.add_argument("--test_limit", type=int, default=100)
    parser.add_argument("--model_half", action="store_true")
    parser.add_argument("--few_shot_prompting", action="store_true")
    args = parser.parse_args()
    main(
        args.model_name,
        args.test_dataset_names,
        args.test_limit,
        args.model_half,
        args.few_shot_prompting,
    )
