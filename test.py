import torch
from torchvision.models.swin_transformer import SwinTransformerBlock

data = torch.randn((10, 8, 16))

block = SwinTransformerBlock(dim=8, num_heads=2, window_size=4, shift_size=2, mlp_ratio=4.0)

out = block(data)

print(out)