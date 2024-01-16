# FlexGen

## Demo

```powershell
git clone https://github.com/FMInference/FlexGen.git || (cd .\FlexGen && git pull)
conda env create -n flexgen -f environment.yaml
conda activate flexgen
conda devlop .

python -m flexgen.flex_opt --model facebook/opt-1.3b

python .\FlexGen\flexgen\apps\completion.py --model facebook/opt-1.3b

cd .\Flexgen
git checkout 9d888e5
python .\flexgen\apps\chatbot.py --model facebook/opt-1.3b
```
