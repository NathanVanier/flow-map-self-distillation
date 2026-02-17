# Learning Flow Maps via Self-Distillation

Reproduction (simplified) of:

Boffi et al., 2025  
"How to Build a Consistency Model: Learning Flow Maps via Self-Distillation"

---

## Overview

Flow maps (consistency models) learn the solution operator of the probability flow ODE,
allowing accelerated sampling without numerical integration.

This project implements a simplified version of:

- Lagrangian Self-Distillation (LSD)
- Progressive Self-Distillation (PSD)

based on the tangent condition and flow map characterization
introduced in the paper.

---

## Mathematical Background

We parameterize the flow map as:

X_{s,t}(x) = x + (t - s) v_{s,t}(x)

and train using:

L_total = L_b + L_D

Where:

- L_b enforces the tangent condition on the diagonal (s = t)
- L_D enforces one of:
  - Lagrangian condition
  - Eulerian condition
  - Semigroup condition

See Proposition 2.3 in the paper.

---

## Implementation Details

- Framework: PyTorch
- Architecture: <MLP / UNet / etc>
- Time sampling: mixture η U_d + (1-η) U_od
- Stopgradient used to stabilize training
- Automatic differentiation used for jvp

---

## Experiments

Dataset:
- <Checkerboard / CIFAR / toy example>

Metrics:
- KL divergence (checker)
- FID (if images)

Results:

| Method | Steps | Metric |
|--------|-------|--------|
| LSD    | 1     | ...    |
| LSD    | 4     | ...    |

---

## How to Run

```bash
pip install -r requirements.txt
python src/training.py --config configs/default.yaml
