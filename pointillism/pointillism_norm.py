# probes/pointillism_norm.py

import torch
from data.transforms import norm_cfg


def normalize_like_loader(x: torch.Tensor, *, dataset: str) -> torch.Tensor:
    """
    Match torchvision Normalize(mean,std) in data/transforms.py.

    x:
      - [B,C,H,W] or [C,H,W]
      - expected in raw [0,1] scale (same as ToTensor output)

    returns:
      normalized tensor with same shape
    """
    mean, std = norm_cfg(dataset)
    device = x.device
    dtype = x.dtype

    if x.dim() == 3:
        x4 = x.unsqueeze(0)
    elif x.dim() == 4:
        x4 = x
    else:
        raise ValueError("Expected x as [C,H,W] or [B,C,H,W]")

    c = int(x4.shape[1])
    if c != len(mean):
        raise ValueError(f"Channel mismatch: got C={c}, expected {len(mean)} for dataset={dataset}")

    m = torch.as_tensor(mean, device=device, dtype=dtype).view(1, c, 1, 1)
    s = torch.as_tensor(std, device=device, dtype=dtype).view(1, c, 1, 1)

    y = (x4 - m) / s
    # print("normalize_like_loader called", dataset, x.min().item(), x.max().item())
    return y.squeeze(0) if x.dim() == 3 else y
