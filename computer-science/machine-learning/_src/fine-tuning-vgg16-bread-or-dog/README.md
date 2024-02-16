---
title: bread-or-dog
app_file: app.py
sdk: gradio
sdk_version: 4.19.0
---
# Fine-Tuning VGG16 to detect bread or dog

See <https://huggingface.co/spaces/xhiroga/bread-or-dog>.

## Prerequisites

```powershell
conda env create -f environment.yml
conda activate fine-tuning-vgg16-bread-or-dog
Invoke-WebRequest -Uri https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json -OutFile ./data/imagenet-simple-labels.json
```

## Demo

```powershell
# Before Fine-Tuning
python vgg16.py

# After Fine-Tunin
python app.py
```

## Deploy

```powershell
gradio deploy
```

## References

- [PyTorchの学習済みモデルで画像分類（VGG, ResNetなど） | note.nkmk.me](https://note.nkmk.me/python-pytorch-pretrained-models-image-classification/)
- [PyTorchによるファインチューニングの実装 - 機械学習ともろもろ](https://venoda.hatenablog.com/entry/2020/10/18/014516)
