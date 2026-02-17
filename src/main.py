# main.py
import torch
from models import VNet
from training import train_lsd
from sampling import sample_steps
from utils import plot_checker_samples, sample_checkerboard
from utils import kl_histogram_2d

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- model ----
vnet = VNet().to(device)

# ---- train ----
vnet = train_lsd(
    vnet,
    steps=30_000,
    batch=2048,
    device=device
)

# ---- generate samples ----
x_gen = sample_steps(vnet, n_samples=50_000, n_steps=1, device=device)
x_true = sample_checkerboard(50_000, device=device)

# ---- plot ----
plot_checker_samples(x_gen, title="Generated (1-step)")
plot_checker_samples(x_true, title="True distribution")

for n in [1, 2, 4, 8, 16]:
    x_gen = sample_steps(vnet, n_samples=50_000, n_steps=n, device=device)
    plot_checker_samples(x_gen, title=f"{n} steps")

# ---- compute KL ----
kl = kl_histogram_2d(x_true, x_gen)
print("KL divergence:", kl)