import json
import os

import google.generativeai as genai
import torch
from tqdm import tqdm
from datasets import dataset_dict, load_dataset
from dotenv import load_dotenv
from peft import PeftMixedModel, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import TypedDict


class Result(TypedDict):
    task_id: int | None
    input: str
    output: str
    eval_aspect: str


class TaskScore(TypedDict):
    task_id: int | None
    score: int


class Evaluation(TypedDict):
    task_id: int | None
    input: str
    output: str
    eval_aspect: str
    score: int


load_dotenv()

# Get Gemini API Key from https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


DEFAULT_EVALUATION_ASPECT = "Evaluate this" # TODO


def validate(
    model: PeftModel | PeftMixedModel,
    tokenizer: AutoTokenizer,
    ds: dataset_dict.Dataset,
    half: bool = False,
    limit: int = None,
) -> list[Result]:
    results = []
    
    test_data = ds if limit is None else ds.select(range(limit))
    for data in tqdm(test_data):
        input = data["input"]

        prompt = f"""\
### 指示
{input}
### 回答
"""

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

        results.append(
            {
                "eval_aspect": DEFAULT_EVALUATION_ASPECT,
                **data,
                "output": output,
            }
        )

    return results


def evaluate(results: list[Result], batch_size: int = 10) -> list[Evaluation]:
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    evaluations = []
    for i in tqdm(range(0, len(results), batch_size)):
        batch_results = results[i : i + batch_size]

        # TODO: からあげさんのプロンプトのエッセンスを取り入れる。
        prompts = [
            f"Evaluate the following result and provide a score between 0 and 5:\nQuestion: {result['input']}\nAnswer: {result['output']}\nEvaluation Aspect: {result['eval_aspect']}"
            for result in batch_results
        ]

        response = model.generate_content(
            prompts,
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


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("models/llm-jp-3-1-8b-finetune", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("models/llm-jp-3-1-8b-finetune", trust_remote_code=True)
    ds = load_dataset("elyza/ELYZA-tasks-100")
    results = validate(model, tokenizer, ds["test"], limit=1)
    evaluations = evaluate(results, 10)
    averagt_score = sum(evaluation["score"] for evaluation in evaluations) / len(
        evaluations
    )
    print(f"{evaluations=}, {averagt_score=}")
    assert 2 < averagt_score < 5
