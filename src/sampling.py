import torch
from losses import X_hat
from utils import sample_base_gaussian


@torch.no_grad()
def sample_steps(vnet, n_samples=50_000, n_steps=1, device="cpu"):
    vnet.eval()

    x = sample_base_gaussian(n_samples, device=device)
    ts = torch.linspace(0, 1, n_steps + 1, device=device)

    for i in range(n_steps):
        s = torch.full((n_samples,), ts[i].item(), device=device)
        t = torch.full((n_samples,), ts[i + 1].item(), device=device)
        x = X_hat(vnet, x, s, t)

    return x