# Get Started | mmpose

## init

See [Installation](https://mmpose.readthedocs.io/en/latest/installation.html).

```shell
python3.8 -m venv .venv
source .venv/bin/activate
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

## demo

See [2d_hand_demo.md](https://github.com/open-mmlab/mmpose/blob/main/demo/docs/en/2d_hand_demo.md).

```shell
git clone https://github.com/open-mmlab/mmpose
cd mmpose
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input tests/data/onehand10k/9.jpg \
    --show --draw-heatmap
```
