# briaai/RMBG

## Prerequisites

```powershell
kaggle datasets download -d mikoajkolman/pokemon-images-first-generation17000-files -p "data/" -q
# unzip
```

## briaai/RMBG-1.4

```powershell
git clone https://huggingface.co/briaai/RMBG-1.4
uv sync
```

### Multiprocessing

```powershell
uv run src/rmbg/remove_background_multiprocess.py
# 途中で実行を止めたが、200件に対して1時間以上かかるようになってしまった。
# GPU利用率は0%と100%を往復しており、処理時間は徐々に長くなっていった。明らかにおかしい。
```
