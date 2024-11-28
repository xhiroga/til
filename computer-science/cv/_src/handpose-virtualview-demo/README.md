---
title: handpose-virtualview-demo
app_file: app.py
sdk: gradio
sdk_version: 4.44.1
---

> [!WARNING]
> I tried my best to get this project working, but I couldn't resolve all the issues. Please be aware that there might be unresolved problems when trying to run this project.

```shell
make
# Edit handpose-virtualview/requirements.txt that replace matplotlib==3.5.1 to 3.6.0
# to avoid AttributeError: module '_tkinter' has no attribute '__file__'
uv sync --index-strategy unsafe-best-match
# Edit .venv/lib/python3.9/site-packages/torch/utils/cpp_extension.py#L1534 to append '8.9' or your CUDA version.
cd handpose-virtualview/ops/cuda
uv run python setup.py build_ext --inplace
cd ../../..
export PYTHONPATH=$PYTHONPATH:$(pwd)/handpose-virtualview/ops/cuda
uv run python handpose-virtualview/train_a2j.py
uv run python app.py
```
