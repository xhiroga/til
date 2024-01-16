---
title: chiikawa-yonezu
app_file: app.py
sdk: gradio
sdk_version: 4.13.0
---

```powershell
conda env create -f environment.yml
conda activate chiikawa-yonezu
pip install fugashi ipadic
```

## Run gradio

```powershell
conda activate chiikawa-yonezu
python app.py
# or
conda run -n chiikawa-yonezu python app.py # not recommended because standard output is not displayed
```

## Deploy to gradio

```powershell
conda activate chiikawa-yonezu
gradio deploy
```
