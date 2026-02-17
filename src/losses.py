import torch
import torch.nn.functional as F


def interpolant(x0, x1, t):
    if t.dim() == 1:
        t = t[:, None]
    It = (1 - t) * x0 + t * x1
    dIt = (x1 - x0)
    return It, dIt


def X_hat(vnet, x, s, t):
    return x + (t - s)[:, None] * vnet(x, s, t)


def dX_dt_hat(vnet, x, s, t):
    t_req = t.clone().detach().requires_grad_(True)

    X = X_hat(vnet, x, s, t_req)

    grads = []
    for k in range(X.shape[1]):
        g = torch.autograd.grad(
            outputs=X[:, k].sum(),
            inputs=t_req,
            create_graph=True,
            retain_graph=True
        )[0]
        grads.append(g[:, None])

    return torch.cat(grads, dim=1)


def lsd_loss(vnet, x0, x1, s, t):
    Is, _ = interpolant(x0, x1, s)

    Xst = X_hat(vnet, Is, s, t)
    dXdt = dX_dt_hat(vnet, Is, s, t)

    vt = vnet(Xst.detach(), t, t)

    return F.mse_loss(dXdt, vt)


def psd_midpoint_loss(vnet, Is, s, t):
    u = 0.5 * (s + t)

    Xsu = X_hat(vnet, Is, s, u)
    Xut = X_hat(vnet, Xsu, u, t)
    Xst = X_hat(vnet, Is, s, t)

    with torch.no_grad():
        teacher = Xut

    return F.mse_loss(Xst, teacher)