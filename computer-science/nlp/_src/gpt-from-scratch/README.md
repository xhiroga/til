# [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)

## Demo

```shell
conda env create -f gpt-from-scratch.yml
conda activate gpt-from-scratch
````

### Encoder

```shell
cd picoGPT
python
```

```python
from utils import load_encoder_hparams_and_params
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
ids = encoder.encode("Not all heros wear capes.")
id
[encoder.decoder[i] for i in ids]
# ['Not', 'Ġall', 'Ġhero', 's', 'Ġwear', 'Ġcap', 'es', '.']

encoder.encode("一日一善")
# [31660, 33768, 98, 31660, 161, 244, 226]
[encoder.decoder[i] for i in encoder.encode("一日一善")]
# ['ä¸Ģ', 'æĹ', '¥', 'ä¸Ģ', 'å', 'ĸ', 'Ħ']

# 流し読みだけど、GPT-2のBPEだと日本語の語彙が少なすぎて不適切なEncodeがされる、ということらしい。[tanreinama/Japanese-BPEEncoder](https://github.com/tanreinama/Japanese-BPEEncoder)
```

### Hyperparameters

```python
hparams
# {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
```

