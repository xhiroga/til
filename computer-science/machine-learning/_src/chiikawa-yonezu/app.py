from pprint import pprint

import gradio as gr
import torch

from safetensors import safe_open
from transformers import BertTokenizer

from utils.ClassifierModel import ClassifierModel


def _classify_text(text, model, device, tokenizer, max_length=20):
    """
    テキストが、'ちいかわ' と '米津玄師' のどちらに該当するかの確率を出力する。
    """

    # テキストをトークナイズし、PyTorchのテンソルに変換
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    pprint(f"inputs: {inputs}")

    # モデルの推論
    model.eval()
    with torch.no_grad():
        outputs = model(
            inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
        )
        pprint(f"outputs: {outputs}")
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # 確率の取得
    chiikawa_prob = probabilities[0][0].item()
    yonezu_prob = probabilities[0][1].item()

    return chiikawa_prob, yonezu_prob


def classify_text(text):
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    pprint(f"device: {device}")

    model_save_path = "models/model.safetensors"
    tensors = {}
    with safe_open(model_save_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    inference_model: torch.nn.Module = ClassifierModel().to(device)
    inference_model.load_state_dict(tensors)

    tokenizer = BertTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )
    chii_prob, yone_prob = _classify_text(text, inference_model, device, tokenizer)
    return {"ちいかわ": chii_prob, "米津玄師": yone_prob}


demo = gr.Interface(
    fn=classify_text,
    inputs="textbox",
    outputs="label",
    examples=[
        "炊き立て・・・・ってコト！？",
        "晴れた空に種を蒔こう",
    ],
)

demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
