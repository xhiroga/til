
```shell
make
# uv add numpy pillow
# uv add torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index https://download.pytorch.org/whl --index-strategy unsafe-best-match
# uv add tqdm matplotlib scipy opencv-python pycocotools --index-strategy unsafe-best-match
uv sync --index-strategy unsafe-best-match

cd A2J-Transformer/dab_deformable_detr/ops
sh make.sh
```
