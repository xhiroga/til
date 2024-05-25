---
title: pokemon-palworld
app_file: app.py
sdk: gradio
sdk_version: 4.17.0
---
# パルワールドで最もポケモンに似ているパルは何か？

## Prerequisites

```powershell
conda env create -f environment.yml
conda activate pokemon-palworld-v2

git clone https://huggingface.co/briaai/RMBG-1.4
cd RMBG-1.4/
pip install -r requirements.txt
```

## Development

### Preprocessing & Train

open notebooks.

### Test
```powershell
python -m unittest discover tests
```

## Run gradio on local

```powershell
python app.py
```

## Deploy to gradio

```powershell
conda activate pokemon-pal
gradio deploy
```
