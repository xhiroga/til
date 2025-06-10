# NVIDIA

NVIDIAのGPU製品はBrandとArchitectureで分類される。いずれも`nvidia-smi`で確認できる。

```console
$ nvidia-smi -q

==============NVSMI LOG==============

Timestamp                                 : Tue Jun 10 18:10:36 2025
Driver Version                            : 572.47
CUDA Version                              : 12.8

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Product Name                          : NVIDIA GeForce RTX 4090
    Product Brand                         : GeForce
    Product Architecture                  : Ada Lovelace
...
```

<details>
<summary>L40S</summary>

```console
nvidia-smi -q

==============NVSMI LOG==============

Timestamp                                 : Tue Jun 10 09:46:16 2025
Driver Version                            : 550.127.05
CUDA Version                              : 12.8

Attached GPUs                             : 1
GPU 00000000:81:00.0
    Product Name                          : NVIDIA L40S
    Product Brand                         : NVIDIA
    Product Architecture                  : Ada Lovelace
```
</details>

## Brand

|ブランド|セグメント|製品例|命名規則|
|---|---|---|---|
|GeForce[^geforce]|ゲーミング|RTX 4090, RTX 4080, GTX 1660|RTX/GTX XXYY（XX:世代、YY:性能ランク）|
|NVIDIA Data Center GPU（旧Tesla）[^datacenter-gpu]|データセンター|H100, A100, L40S / Tesla V100, Tesla T4|型番のみ（2020年5月以降）|
|NVIDIA RTX[^rtx-pro]|プロフェッショナル|RTX 6000 Ada, RTX A6000|RTX + 数字 + (世代名)|
|Quadro[^quadro]|プロフェッショナル（旧）|Quadro RTX 8000|Quadro + RTX/型番|
|NVIDIA DRIVE[^drive]|自動車|DRIVE Orin, DRIVE Thor|DRIVE + 製品名|

[^geforce]: [GeForce Graphics Cards](https://www.nvidia.com/en-us/geforce/)
[^datacenter-gpu]: [NVIDIA Data Center](https://www.nvidia.com/en-us/data-center/) - 2020年5月以降、Teslaブランドを廃止。Tesla車との混同回避のため型番のみに変更
[^rtx-pro]: [NVIDIA RTX Professional](https://www.nvidia.com/en-us/design-visualization/rtx/)
[^quadro]: 2020年以降NVIDIA RTXブランドに移行
[^drive]: [NVIDIA DRIVE Platform](https://www.nvidia.com/en-us/self-driving-cars/)

## Architecture

|アーキテクチャ|発表年月|Compute Capability|主な製品|主な機能|命名由来|
|---|---|---|---|---|---|
|Blackwell[^blackwell]|2024年3月|10.0, 12.0|B100 (700W), B200 (1000W), GB200|第5世代Tensor Core、FP4/FP8演算、208B トランジスタ|デイヴィッド・ブラックウェル（米統計学者）|
|Ada Lovelace[^ada]|2022年9月|8.9|RTX 40シリーズ、L40S|第3世代RT Core、AV1エンコード|エイダ・ラブレス（英数学者）|
|Hopper[^hopper]|2022年3月|9.0|H100, H200|第4世代Tensor Core、Transformer Engine|グレース・ホッパー（米計算機科学者）|
|Ampere[^ampere]|2020年5月|8.0, 8.6, 8.7|A100, A40, RTX 30シリーズ|第3世代Tensor Core、構造的スパース性|アンドレ＝マリ・アンペール（仏物理学者）|

### Compute Capability について

Compute Capabilityは、GPUの機能セットとアーキテクチャ世代を示す数値です：
- **メジャー番号**（小数点前）: アーキテクチャの大きな変更を示す
- **マイナー番号**（小数点後）: 同一アーキテクチャ内の改良版を示す

例：
- 10.0, 12.0 = Blackwell（B100, B200など）
- 9.0 = Hopper（H100）
- 8.9 = Ada Lovelace（RTX 4090など）
- 8.6 = Ampere（RTX 3090, A100など）

確認方法：
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

[^blackwell]: [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) - GTC 2024（2024年3月18日）で発表。デュアルダイ設計、CUDA 12.8以降が必要
[^ada]: [Ada Lovelace Architecture](https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/) - 2022年9月20日発表
[^hopper]: [Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) - GTC 2022で発表
[^ampere]: [Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/) - A100は2020年5月発表、RTX 30シリーズは2020年9月発表
