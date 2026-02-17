import torch
import numpy as np
import matplotlib.pyplot as plt


def sample_checkerboard(n, n_cells=4, cell_size=1.0, jitter=0.0, device="cpu"):
    coords = []
    for i in range(n_cells):
        for j in range(n_cells):
            if (i + j) % 2 == 0:
                coords.append((i, j))
    coords = torch.tensor(coords, device=device, dtype=torch.float32)

    idx = torch.randint(0, coords.shape[0], (n,), device=device)
    cells = coords[idx]

    noise = torch.rand(n, 2, device=device)
    samples = (cells + noise) * cell_size
    samples = samples - (n_cells * cell_size) / 2.0

    if jitter > 0:
        samples += jitter * torch.randn_like(samples)

    return samples


def sample_base_gaussian(n, device="cpu"):
    return torch.randn(n, 2, device=device)


def plot_checker_samples(samples, title="", lim=2.5, s=1):
    x = samples.detach().cpu()
    plt.figure(figsize=(4, 4))
    plt.scatter(x[:, 0], x[:, 1], s=s, alpha=0.5)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", "box")
    plt.title(title)
    plt.show()


def kl_histogram_2d(x_true, x_gen, bins=50, lim=2.5, eps=1e-8):
    x_true = x_true.cpu().numpy()
    x_gen = x_gen.cpu().numpy()

    edges = np.linspace(-lim, lim, bins + 1)

    p_hist, _, _ = np.histogram2d(
        x_true[:, 0], x_true[:, 1],
        bins=[edges, edges],
        density=True
    )

    q_hist, _, _ = np.histogram2d(
        x_gen[:, 0], x_gen[:, 1],
        bins=[edges, edges],
        density=True
    )

    p = p_hist + eps
    q = q_hist + eps

    return np.sum(p * np.log(p / q))