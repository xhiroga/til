# FlexGen

## Demo

```powershell
git clone https://github.com/FMInference/FlexGen.git || (cd ./FlexGen && git pull)
conda create -n flexgen -f environment.yaml
conda activate flexgen
conda devlop .

python -m flexgen.flex_opt --model facebook/opt-1.3b
```
