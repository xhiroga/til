import csv

import torch
from peft import PeftMixedModel, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, logging


logger = logging.get_logger("transformers")


class DynamicFewShotPromptingEngineByLLM:
    def __init__(
        self,
        categorized_path: str,
        model: PeftModel | PeftMixedModel,
        tokenizer: AutoTokenizer,
    ) -> None:
        categorized = []
        with open(categorized_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                categorized.append(row)
        logger.info(f"Loaded {len(categorized)} categorized tasks")
        logger.debug(f"{categorized[:5]=}")
        self.categorized = categorized
        self.categories = set(row["category"] for row in categorized)

        self.categorize_prompt_template = """\
Please categorize the following task into one of the following categories: \
""" + ", ".join(sorted(self.categories)) + """

Note: Do NOT answer the task itself. ONLY Answer the category of the task.

# Task (Example)
あなたは、友人から「最近物忘れがひどくて困っている」と相談を受けました。どのような返事をしますか？

# Category (Example)
会話:親切

# Task
{}

# Category
"""
        logger.debug(f"{self.categorize_prompt_template=}")

        self.model = model
        self.tokenizer = tokenizer


    def categorize(self, task: str) -> str:
        tokenized_input = self.tokenizer.encode(
            self.categorize_prompt_template.format(task), add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)
        attention_mask = torch.ones_like(tokenized_input)

        with torch.no_grad():
            outputs = self.model.generate(
                tokenized_input,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]
        output = self.tokenizer.decode(
            outputs[tokenized_input.size(1):], skip_special_tokens=True
        )

        category = output.strip()
        logger.debug(f"{category=}")

        return category


if __name__ == "__main__":
    logging.set_verbosity_debug()
    categorized_path = "data/ELYZA-tasks-100-categorized - categorized.csv"
    model = AutoModelForCausalLM.from_pretrained(
        "models/llm-jp/llm-jp-3-1.8b", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "models/llm-jp/llm-jp-3-1.8b", trust_remote_code=True
    )
    engine = DynamicFewShotPromptingEngineByLLM(categorized_path, model, tokenizer)
    test_instruction = """\
"次の言葉を順に全て使って、1つの文を書いてください

努力、未来、a、beautiful star
"""
    print(engine.categorize(test_instruction))
