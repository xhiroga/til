# ComfyUI

## ComfyUI-ManagerへのCustom Nodeの追加

実験は[`_src/manager`](_src/manager)を参照。

1. ComfyUI-Managerをインストール
2. ComfyUI/custom_nodes/ComfyUI-Manager/custom-node-list.json を編集
3. ComfyUI Managerの設定を開き、DB: Channel → Local に変更
4. Custom Nodeのインストールを実行

### 編集の例

```json
{
    "author": "xhiroga",
    "title": "ComfyUI-FramePackWrapper_PlusOne",
    "id": "comfyui-framepackwrapper-plusone",
    "reference": "https://github.com/xhiroga/ComfyUI-FramePackWrapper_PlusOne",
    "files": [
        "https://github.com/xhiroga/ComfyUI-FramePackWrapper_PlusOne"
    ],
    "install_type": "git-clone",
    "description": "ComfyUI extension for FramePack, supporting 1-frame inferences."
}
```
