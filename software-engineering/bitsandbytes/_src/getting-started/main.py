from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "gpt2"  # 小さめの例を使うなら gpt2 などで OK
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 量子化設定：8ビットモードでロードする
bnb_config = BitsAndBytesConfig(
    load_in_8bit = True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = "auto"  # GPU があれば自動で割り当て
)

# 推論例
input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
