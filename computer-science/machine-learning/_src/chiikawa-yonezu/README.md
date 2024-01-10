# ちいかわか米津玄師か分類タスク

```powershell
conda create -f environment.yml
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
