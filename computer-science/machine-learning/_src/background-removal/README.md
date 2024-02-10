# 背景透過ソフトウェアの比較

## Pre-requisites

```powershell
kaggle datasets download -d mikoajkolman/pokemon-images-first-generation17000-files -p "data/" -q
# unzip
```

## briaai/RMBG-1.4

```powershell
conda env create -f environment.yml
conda activate background-removal

git clone https://huggingface.co/briaai/RMBG-1.4
cd RMBG-1.4/
pip install -r requirements.txt
```
