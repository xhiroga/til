# Packaging musubi-tuner

## Install

### pip

```powershell
winget install -e --id Python.Python.3.10
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/xhiroga/musubi-tuner-xhiroga.git@feat/package

.\run.ps1
```

### uv

```powershell
uv init --python 3.10
.venv\Scripts\activate
uv add git+https://github.com/xhiroga/musubi-tuner-xhiroga.git@feat/package

.\run.ps1
```
