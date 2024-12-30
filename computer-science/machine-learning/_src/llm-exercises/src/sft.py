import os
import argparse
import bitsandbytes as bnb
import torch
import wandb
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from trl import SFTTrainer

from instruction_datasets import INSTRUCTION_DATASETS
from utils import infer

load_dotenv()
logging.set_verbosity_info()

HF_TOKEN = os.environ.get("HF_TOKEN")
assert os.environ.get("WANDB_ENTITY")
assert os.environ.get("WANDB_PROJECT")

def main(
    model_name="llm-jp/llm-jp-3-13b",
    test_dataset_names=["elyza/ELYZA-tasks-100", "elyza-tasks-100-TV_0"],
    test_limit=100,
    model_half=False,
    few_shot_prompting=False,
):
    config = {
        "base_model_id": model_name,
        "train": {"datasets": ["ichikara-instruction-all"]},
        "test": {
            "datasets": test_dataset_names,
            # few_shot_promptingをサポートする場合はここでプロンプトを変更する
            "prompt": """\
### 指示
以下の評価基準を厳守して回答してください:
- 必要な箇所でアイデア数を守る
- 指示されたプロットや要件を満たす
- 重複やズレがないよう注意

{}

### 回答
""",
            "limit": test_limit,
            "model_half": model_half,
            "max_new_tokens": 200,
        },
    }

    run = wandb.init(config=config, job_type="sft")
    run_name = wandb.run.name

    new_model_id = f"{config['base_model_id'].replace('.', '-')}-finetune-{run_name}"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_id"], quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_id"], trust_remote_code=True)

    def find_all_linear_names(model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[-1])
        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
        return list(lora_module_names)

    modules = find_all_linear_names(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )

    model = get_peft_model(model, peft_config)
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        # 学習用のデータセットのフォーマット例
        # "input"と"output"があることが前提
        input = examples["input"]
        output = examples["output"]
        text = config["test"]["prompt"].format(input) + output + EOS_TOKEN
        return {"formatted_text": text}

    dataset = INSTRUCTION_DATASETS[config["train"]["datasets"][0]]()
    dataset = dataset.map(formatting_prompts_func, num_proc=8)

    # 適宜train/eval splitを行う
    # dataset = dataset["train"].train_test_split(test_size=0.1)

    training_arguments = TrainingArguments(
        output_dir=f"models/{new_model_id}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        logging_strategy="steps",
        logging_steps=10,
        warmup_steps=10,
        save_steps=100,
        save_total_limit=2,
        max_steps=-1,
        learning_rate=5e-5,
        fp16=True,
        bf16=False,
        seed=3407,
        group_by_length=True,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=1024,
        dataset_text_field="formatted_text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    model.config.use_cache = False
    tokenizer.pad_token = tokenizer.eos_token
    trainer.train()

    inference = infer(
        model,
        tokenizer,
        config["test"]["datasets"],
        new_model_id,
        run_name=run_name,
        test_prompt=config["test"]["prompt"],
        test_limit=config["test"]["limit"],
        model_half=config["test"]["model_half"],
        max_new_tokens=config["test"]["max_new_tokens"],
    )
    wandb.log(inference)

    model.push_to_hub(
        new_model_id, token=HF_TOKEN, private=True
    )
    tokenizer.push_to_hub(
        new_model_id, token=HF_TOKEN, private=True
    )

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llm-jp/llm-jp-3-13b")
    parser.add_argument("--test_dataset_names", type=str, nargs="+", default=["elyza/ELYZA-tasks-100", "elyza-tasks-100-TV_0"])
    parser.add_argument("--test_limit", type=int, default=100)
    parser.add_argument("--model_half", action="store_true")
    parser.add_argument("--few_shot_prompting", action="store_true")
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        test_dataset_names=args.test_dataset_names,
        test_limit=args.test_limit,
        model_half=args.model_half,
        few_shot_prompting=args.few_shot_prompting,
    )
