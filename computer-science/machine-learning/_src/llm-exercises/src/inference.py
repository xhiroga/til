import argparse
import os

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from instruction_datasets import INSTRUCTION_DATASETS
from utils import infer

assert os.environ.get("WANDB_ENTITY")
assert os.environ.get("WANDB_PROJECT")
model_dir = "models"

load_dotenv()


def main(
    model_name: str,
    test_dataset_names: list[str] = ["elyza/ELYZA-tasks-100", "elyza-tasks-100-TV_0"],
    few_shot_prompting_dataset_name: str | None = None,
    few_shot_prompting_limit: int | None = None,
):
    if few_shot_prompting_dataset_name and few_shot_prompting_limit:
        ds = INSTRUCTION_DATASETS[few_shot_prompting_dataset_name]().select(
            range(few_shot_prompting_limit)
        )
        test_prompt = (
            """\
200字程度で簡潔に回答してください。\
"""
            + "\n".join(
                [
                    f"# 指示（例）\n{task['input']}\n# 回答（例）\n{task['output']}\n"
                    for task in ds
                ]
            )
            + """
# 指示
{}
# 回答
"""
        )
    else:
        test_prompt = """\
200字程度で簡潔に回答してください。

# 指示
{}
# 回答
"""

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
    parser.add_argument("--test_dataset_names", type=list, nargs="+")
    parser.add_argument("--few_shot_prompting_dataset_name", type=str)
    parser.add_argument("--few_shot_prompting_limit", type=int)
    args = parser.parse_args()
    main(
        args.model_name,
        args.test_dataset_names,
        args.few_shot_prompting_dataset_name,
        args.few_shot_prompting_limit,
    )
