import torch


def structural_causal_mask(B: int, L: int, device: torch.device) -> torch.Tensor:
    mask_shape = [B, 1, L, L]
    mask = torch.triu(
        torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1
    )
    return mask
