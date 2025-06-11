#!/usr/bin/env python
"""
FP8 対応チェック + 簡易デモ

* Hopper (H100) / Blackwell (B100/B200) 世代など CC 9.0 以上の GPU で
  Transformer Engine の fp8_autocast を用いた Linear 演算を試す。
* Transformer Engine が入っていない場合は pip install を案内。
"""

import torch
import importlib.util
import sys

def has_fp8_tensorcore():
    """FP8 Tensor Core 対応の有無と GPU 情報を返す"""
    if not torch.cuda.is_available():
        return False, "CUDA が利用不可です。", None
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor                  # 例: 9.0 → 90
    name = torch.cuda.get_device_name()
    return sm >= 90, name, f"{major}.{minor}"  # SM90 以上を FP8 対応と見なす

def run_fp8_demo():
    """Transformer Engine で実際に FP8 Linear を 1 回実行する"""
    import transformer_engine.pytorch as te

    device = "cuda"
    # 1024×1024 の線形層を作成（重みは FP16）
    linear = te.Linear(1024, 1024, bias=False, params_dtype=torch.float16, device=device)
    x = torch.randn(1, 1024, device=device, dtype=torch.float16)

    # fp8_autocast コンテキスト内で forward
    with te.fp8_autocast(enabled=True):
        y = linear(x)

    print(f"✓ FP8 forward 成功！ 出力 dtype: {y.dtype}")  # y はデフォルトで FP16

def main():
    supported, name, cc = has_fp8_tensorcore()
    if not supported:
        print(f"× {name} (CC {cc}) は FP8 Tensor Core 未対応、または GPU がありません。")
        sys.exit(0)

    print(f"◎ {name} (CC {cc}) は FP8 Tensor Core に対応しています。")

    # Transformer Engine の有無を確認
    if importlib.util.find_spec("transformer_engine") is None:
        print("Transformer Engine が見つかりません。以下でインストールしてください：")
        print("  pip install transformer-engine --extra-index-url https://pypi.nvidia.com")
        sys.exit(0)

    # 実際に FP8 演算を試す
    try:
        run_fp8_demo()
    except Exception as e:
        print("! FP8 演算デモで例外が発生しました:", e)

if __name__ == "__main__":
    main()
