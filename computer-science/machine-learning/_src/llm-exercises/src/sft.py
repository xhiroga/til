import os
from pathlib import Path

import bitsandbytes as bnb
import torch
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

import wandb
from instruction_datasets import INSTRUCTION_DATASETS
from utils import infer

load_dotenv()

logging.set_verbosity_info()

HF_TOKEN = os.environ.get("HF_TOKEN")
assert os.environ.get("WANDB_ENTITY")
assert os.environ.get("WANDB_PROJECT")

config = {
    "base_model_id": "llm-jp/llm-jp-3-1.8b",
    "mode": "sft",
    "train":
        {"datasets": ["ichikara-instruction-all"]},
    "test": {
            "datasets": ["elyza/ELYZA-tasks-100", "elyza-tasks-100-TV_0"],
            "prompt": """\
### 指示
200字程度で簡潔に回答してください。
{}
### 回答
""",
            "limit": None,
            "model_half": False,
            "max_new_tokens": 200,
        },
}

wandb.init(config=config)
run_name = wandb.run.name

new_model_id = f"{config['base_model_id'].replace('.', '-')}-finetune-{run_name}"

"""
bnb_config: 量子化の設定

  - load_in_4bit:
      - 4bit量子化形式でモデルをロード

  - bnb_4bit_quant_type:
      - 量子化の形式を指定

  - bnb_4bit_compute_dtype:
      - 量子化された重みを用いて計算する際のデータ型

"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # nf4は通常のINT4より精度が高く、ニューラルネットワークの分布に最適です
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    config['base_model_id'], quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config['base_model_id'], trust_remote_code=True)

"""
find_all_linear_names: モデル内の4bit量子化線形層を探します。
"""


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # 4bit量子化線形層クラスを指定
    lora_module_names = set()  # ここに取得した線形層を保持します。

    # モデル内の全てのモジュールを探索します
    for name, module in model.named_modules():
        if isinstance(module, cls):  # モジュールが4bit量子化線形層の場合
            names = name.split(
                "."
            )  # モジュールの名前を分割 (ネストされてる際などに対処)
            lora_module_names.add(
                names[0] if len(names) == 1 else names[-1]
            )  # 最下層の名前をlora_module_namesに追加

    # 'lm_head' は16ビット演算の際に除外する必要があるため、lora_module_namesから削除
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)  # lora_module_namesをリストに変換して返します。


modules = find_all_linear_names(model)

"""
peft_config: PEFTの構成設定

  - r
      - LoRA のランク (4, 8, 16 ,32...)
      - 増やすほど学習が捗るが, 過学習のリスクも高まるので注意

  - lora_alpha
      - LoRAのスケーリング係数

  - lora_dropout
      - ドロップアウト率（過学習を防ぐための割合）

  - bias
      - バイアス項の扱い ("none"の場合、LoRAはバイアスを学習しない)

  - task_type
      - タスクタイプ

  - target_modules
      - LoRAを適用するターゲットモジュール (前のコードで特定した層)
"""

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

model = get_peft_model(model, peft_config)

EOS_TOKEN = tokenizer.eos_token  # トークナイザーのEOSトークン（文末トークン）


def formatting_prompts_func(examples):
    input = examples["input"]  # 入力データ
    output = examples["output"]  # 出力データ
    text = config["test"]["prompt"].format(input, output) + EOS_TOKEN  # プロンプトの作成
    return {
        "formatted_text": text,
    }  # 新しいフィールド "formatted_text" を返す


# 各データにフォーマットを適用
dataset = INSTRUCTION_DATASETS[config["train"]["datasets"][0]]()
dataset = dataset.map(
    formatting_prompts_func,
    num_proc=8,  # 並列処理数を指定
)

# TODO: データをtrainデータとtestデータに分割 (test_sizeの比率に)
# dataset = dataset["train"].train_test_split(test_size=0.1)
# dataset

# TODO: tf32 の検討
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
    # TODO: eval_dataset
    peft_config=peft_config,
    max_seq_length=1024,  # TODO: 長くすることを検討
    dataset_text_field="formatted_text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

model.config.use_cache = False  # キャッシュ機能を無効化
tokenizer.pad_token = tokenizer.eos_token
trainer.train()  # トレーニングを実行

# test
inference = infer(
        model,
        tokenizer,
        ["elyza/ELYZA-tasks-100", "elyza-tasks-100-TV_0"],
        new_model_id,
        run_name=run_name,
        test_prompt=config["test"]["prompt"],
        test_limit=config["test"]["limit"],
        model_half=config["test"]["model_half"],
        max_new_tokens=config["test"]["max_new_tokens"],
    )
wandb.log(inference)

wandb.finish()

# モデルとトークナイザーをHugging Faceにアップロード
model.push_to_hub(
    "llm-jp-3-1-8b-finetune", token=HF_TOKEN, private=True
)  # Online saving
tokenizer.push_to_hub(
    "llm-jp-3-1-8b-finetune", token=HF_TOKEN, private=True
)  # Online saving
