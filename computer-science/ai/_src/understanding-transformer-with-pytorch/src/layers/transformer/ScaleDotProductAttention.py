# Copyright 2022 @YadaYuki. All Rights Reserved.

# アテンション機構とは？
# 入力のトークン（単語）の埋め込みベクトルを出力のベクトルに変換する上で、他の単語との関係性の重さも考慮したもの。

import numpy as np
import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        scaler = np.sqrt(self.d_k)
        # matmul: 行列の積(MATrix MULtiplication)
        # TODO: scaler?
        # TODO: Jupyter Notebookで動かすために次のようにしていた。torch.transpose(k, 0, 1)
        attention_weight = torch.matmul(query, torch.transpose(key, 1, 2)) / scaler

        # decoderの場合、未来の情報を見ないようにするためのマスクをかける
        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask == 0, -1e9)

        # 対象のトークンとの類似度（attention_weight）を確率として扱うためにsoftmaxを行い、valueとの加重平均をとって出力値とする。
        attention_weight = torch.softmax(attention_weight, dim=-1)
        return torch.matmul(attention_weight, value)
