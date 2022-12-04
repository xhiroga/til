#/bin/bash

git clone https://github.com/vsymbol/CUTIE
sed -i 's/opencv-python==4.0.0.21/opencv-python==4.3.0.38/g' CUTIE/requirements.txt # 4.0.0.21 not found. I cannot understand the reason.
sed -i 's/tensorflow==1.12.0/https:\/\/storage.googleapis.com\/tensorflow\/mac\/cpu\/tensorflow-1.12.0-py3-none-any.whl/g' CUTIE/requirements.txt   # pip cannot find package automatically
pip install -r CUTIE/requirements.txt
