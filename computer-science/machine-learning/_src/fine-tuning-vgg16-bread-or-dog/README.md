# Fine-Tuning VGG16 to detect bread or dog

## Prerequisites

```powershell
conda env create -f environment.yml
conda activate fine-tuning-vgg16-bread-or-dog
Invoke-WebRequest -Uri https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json -OutFile ./data/imagenet-simple-labels.json
```

## Demo

```powershell
python vgg16.py
```

## References

- [PyTorchの学習済みモデルで画像分類（VGG, ResNetなど） | note.nkmk.me](https://note.nkmk.me/python-pytorch-pretrained-models-image-classification/)
- [PyTorchによるファインチューニングの実装 - 機械学習ともろもろ](https://venoda.hatenablog.com/entry/2020/10/18/014516)
