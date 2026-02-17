import torch
from losses import interpolant, lsd_loss, psd_midpoint_loss
from utils import sample_checkerboard, sample_base_gaussian


def sample_upper_triangle(B, device):
    t = torch.rand(B, device=device)
    s = torch.rand(B, device=device) * t
    return s, t


def train_lsd(vnet, steps=50_000, batch=2048, eta=0.75, lr=2e-4,
              n_cells=4, device="cpu"):

    opt = torch.optim.AdamW(vnet.parameters(), lr=lr, weight_decay=1e-4)

    for it in range(1, steps + 1):
        vnet.train()

        Md = int(batch * eta)
        Mo = batch

        # diagonal part
        x0 = sample_base_gaussian(Md, device=device)
        x1 = sample_checkerboard(Md, n_cells=n_cells, device=device)
        t = torch.rand(Md, device=device)

        Is, dIt = interpolant(x0, x1, t)
        pred = vnet(Is, t, t)
        Lb = torch.mean((pred - dIt) ** 2)

        # off-diagonal LSD
        x0 = sample_base_gaussian(Mo, device=device)
        x1 = sample_checkerboard(Mo, n_cells=n_cells, device=device)
        s, t = sample_upper_triangle(Mo, device)

        Ld = lsd_loss(vnet, x0, x1, s, t)

        loss = Lb + Ld

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 1000 == 0:
            print(f"[{it}] loss={loss.item():.6f}")

    return vnet


def train_psd(vnet, steps=50_000, batch=2048, eta=0.75, lr=2e-4,
              n_cells=4, device="cpu"):

    opt = torch.optim.AdamW(vnet.parameters(), lr=lr, weight_decay=1e-4)

    for it in range(1, steps + 1):
        vnet.train()

        Md = int(batch * eta)
        Mo = batch

        # diagonal
        x0 = sample_base_gaussian(Md, device=device)
        x1 = sample_checkerboard(Md, n_cells=n_cells, device=device)
        t = torch.rand(Md, device=device)

        Is, dIt = interpolant(x0, x1, t)
        pred = vnet(Is, t, t)
        Lb = torch.mean((pred - dIt) ** 2)

        # PSD off-diagonal
        x0 = sample_base_gaussian(Mo, device=device)
        x1 = sample_checkerboard(Mo, n_cells=n_cells, device=device)
        s, t = sample_upper_triangle(Mo, device)

        Is, _ = interpolant(x0, x1, s)
        Ld = psd_midpoint_loss(vnet, Is, s, t)

        loss = Lb + Ld

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 1000 == 0:
            print(f"[{it}] loss={loss.item():.6f}")

    return vnet