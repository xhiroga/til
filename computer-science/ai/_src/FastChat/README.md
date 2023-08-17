# FastChat

```cmd
git clone https://github.com/lm-sys/FastChat || (cd .\FastChat && git pull)

conda env create -f environment.yml
conda activate FastChat
pip install -e .
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/huggingface/transformers

# 2023-04-01時点では法的な懸念を解消しきっておらず、Vicunaのモデルは公開されていない。
python -m fastchat.serve.cli --model-name facebook/opt-1.3b
```

## References

[GitHub - lm-sysFastChat: The release repo for "Vicuna: An Open Chatbot Impressing GPT-4"](https://github.com/lm-sys/FastChat)
