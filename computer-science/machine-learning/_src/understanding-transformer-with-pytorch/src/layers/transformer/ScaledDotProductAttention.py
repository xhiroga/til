# Copyright 2022 @YadaYuki. All Rights Reserved.

import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        q: torch.Tensor,  # target
        k: torch.Tensor,  # source
        v: torch.Tensor,  # source
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        scalar = np.sqrt(self.d_k)
        # matmul: 行列の積(MATrix MULtiplication)
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar
        # decoderの場合、未来の情報を見ないようにするためのマスクをかける
        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            )
        # 対象のトークンとの類似度（attention_weight）を確率として扱うためにsoftmaxを行い、valueとの加重平均をとって出力値とする。
        attention_weight = nn.functional.softmax(attention_weight, dim=2)
        return torch.matmul(attention_weight, v)
