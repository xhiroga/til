---
title: pokemon-palworld
app_file: app.py
sdk: gradio
sdk_version: 4.16.0
---
# パルワールドで最もポケモンに似ているパルは何か？

## Run gradio on local

```powershell
conda env create -f environment.yml
conda activate pokemon-pal
python app.py
```

## Deploy gradio

```powershell
conda activate pokemon-pal
gradio deploy
```
