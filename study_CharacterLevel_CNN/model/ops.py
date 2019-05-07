
import torch
import torch.nn as nn


class Flatten(nn.Module):
    """
    Flatten class
    """

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1) # Resize 텐서의 모양을 변경


class Permute(nn.Module):
    """
    Permute class
    """

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)


# -------------------------
# torch.view : Resize 텐서의 모양을 변경 / 참고 링크 - https://zzsza.github.io/data/2018/02/03/pytorch-1/
# torch.size : 텐서 shape 출력 , torch.size(0) : row len 출력
# torch.permute : 텐서 역행렬 차원 변 / 참고 링크 - https://discuss.pytorch.org/t/permute-elements-of-a-tensor-along-a-dimension/1105/12
# -------------------------

