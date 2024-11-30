import copy
import json
import os

import google.generativeai as genai
import torch
from datasets import dataset_dict, load_dataset
from dotenv import load_dotenv
from peft import PeftMixedModel, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from typing_extensions import TypedDict


class Result(TypedDict):
    task_id: int | None
    input: str
    output: str
    eval_aspect: str | None
    target: str | None


class TaskScore(TypedDict):
    task_id: int | None
    score: int


class Evaluation(TypedDict):
    task_id: int | None
    input: str
    output: str
    eval_aspect: str | None
    target: str | None
    score: int


load_dotenv()

logger = logging.get_logger("transformers")

# Get Gemini API Key from https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

PREREQUISITE_PROMPT = """\
あなたは採点者です。

問題, 採点基準, 回答 が与えられます。
回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
"""

def evaluation_prompt(input: str, output: str, eval_aspect: str | None, target: str | None) -> str:
    return f"""\
回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題: {input}

{f"# 正解例: {target}" if target is not None else ""}

{f"# 採点基準: {eval_aspect}" if eval_aspect is not None else ""}

# 回答: {output}
"""


def test(
    model: PeftModel | PeftMixedModel,
    tokenizer: AutoTokenizer,
    ds: dataset_dict.Dataset,
    limit: int = None,
    model_half: bool = False,
) -> list[Result]:
    logger.info("start testing")

    results = []
    
    test_data = ds if limit is None else ds.select(range(limit))
    for data in tqdm(test_data):
        input = data["input"]

        prompt = f"""\
# 指示
{input}
# 回答
"""

        if model_half:
            model.half()
        tokenized_input = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(model.device)
        attention_mask = torch.ones_like(tokenized_input)

        with torch.no_grad():
            outputs = model.generate(
                tokenized_input,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        output = tokenizer.decode(
            outputs[tokenized_input.size(1) :], skip_special_tokens=True
        )

        result = copy.deepcopy(data)
        result["output"] = output
        if "output" in data:
            result["target"] = data["output"]
        results.append(result)

    return results


def evaluate(results: list[Result], batch_size: int = 10) -> list[Evaluation]:
    logger.info("start evaluating")
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    evaluations = []
    for i in tqdm(range(0, len(results), batch_size)):
        batch_results = results[i : i + batch_size]

        prompts = [
            evaluation_prompt(result["input"], result["output"], result["eval_aspect"], result.get("sample_output"))
            for result in batch_results
        ]

        response = model.generate_content(
            [PREREQUISITE_PROMPT] + prompts,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=list[TaskScore]
            ),
        )
        scores = json.loads(response.parts[0].text)

        for result, score in zip(batch_results, scores):
            evaluations.append(
                {
                    **result,
                    "score": score["score"],
                }
            )

    return evaluations


# LLM講座として必要なのは、task_id と output のみ
def save_results(results: list[Result], jsonl_prefix: str):
    with open(f"output/{jsonl_prefix}-outputs.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            json.dump(
                result, f, ensure_ascii=False
            )  # ensure_ascii=False for handling non-ASCII characters
            f.write("\n")


if __name__ == "__main__":
    logging.set_verbosity_info()
    model = AutoModelForCausalLM.from_pretrained("models/llm-jp-3-1-8b-finetune", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("models/llm-jp-3-1-8b-finetune", trust_remote_code=True)
    ds = load_dataset("elyza/ELYZA-tasks-100")
    results = test(model, tokenizer, ds["test"], limit=5, model_half=True)
    evaluations = evaluate(results, 10)
    averagt_score = sum(evaluation["score"] for evaluation in evaluations) / len(
        evaluations
    )
    print(f"{evaluations=}, {averagt_score=}")
    assert 2 < averagt_score < 5
