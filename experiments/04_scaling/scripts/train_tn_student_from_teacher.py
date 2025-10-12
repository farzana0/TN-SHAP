#!/usr/bin/env python3
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import math
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim

# ---- import your model ----
# from src.tntree_model import BinaryTensorTree, make_balanced_binary_tensor_tree
from src.tntree_model import BinaryTensorTree

# ---------- Settings ----------
torch.set_default_dtype(torch.float64)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2711
N_FEATURES = 50
LEAF_DIMS = [2] * N_FEATURES         # [x_i, 1] pattern (your model auto-handles bias when matrix)
RANK = 4                              # modest rank; higher if you like, but 4 is very stable
OUT_DIM = 1
N_TRAIN = 20000                       # plenty to drive MSE to ~0
BATCH = 1024
LR = 5e-3
EPOCHS = 400
CLIP = 1.0

g = torch.Generator(device=DEVICE).manual_seed(SEED)

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute R^2 (coefficient of determination)."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-18))
# ---------- Helpers ----------
def _delta_like_core(shape, gen, scale=1.0):
    """
    Build a near-identity / delta-like tensor:
    - For leaf cores [phys, rin]: random with small std and scaled so output magnitudes are O(1).
    - For internal cores [rl, rr, rin]: initialize close to a “copy” along matching indices.
    - For root if out_dim>1: we still keep it well-conditioned.

    Returns torch.Tensor on DEVICE.
    """
    t = torch.zeros(*shape, device=DEVICE)
    if len(shape) == 2:
        # leaf: [phys=2, rin]
        # make it so input [:,0]=x flows, [:,1]=1 flows, balanced
        # Start small random, then add a diagonal-ish mapping on last axis
        std = 0.02
        t.normal_(mean=0.0, std=std, generator=gen)
        # Encourage pass-through on both channels to prevent dead paths
        # If rin==1, just map both phys to that single bond reasonably.
        rin = shape[1]
        if rin == 1:
            t[0, 0] += 0.7 * scale  # x -> bond
            t[1, 0] += 0.7 * scale  # 1 -> bond
        else:
            k = min(2, rin)
            for i in range(k):
                t[i, i] += 0.7 * scale
            if rin > 2:
                # spread remaining columns with small bias to avoid rank deficiency
                for j in range(2, rin):
                    t[1, j] += 0.3 * scale
    elif len(shape) == 3:
        rl, rr, rin = shape
        std = 0.02 / math.sqrt(rl + rr + rin)
        t.normal_(mean=0.0, std=std, generator=gen)
        # encourage "copy" along a common index up to min ranks
        k = min(rl, rr, rin)
        for i in range(k):
            t[i, i, i] += 0.8 * scale
    else:
        # root with out_dim==1 has shape [rl, rr], but in our class root when out_dim==1 is [rl, rr] (2D)
        # If out_dim>1 then [rl, rr, out]; treat like small random + mild structure
        std = 0.02 / math.sqrt(sum(shape))
        t.normal_(mean=0.0, std=std, generator=gen)
        if len(shape) == 2:
            rl, rr = shape
            k = min(rl, rr)
            for i in range(k):
                t[i, i] += 0.8 * scale
        elif len(shape) == 3:
            rl, rr, out = shape
            k = min(rl, rr)
            for i in range(k):
                t[i, i, 0] += 0.8 * scale
    return t


def make_teacher_tree(n_features: int, rank: int, out_dim: int = 1, seed: int = 0) -> BinaryTensorTree:
    torch.manual_seed(seed)
    # Build teacher with safe cores
    teacher = BinaryTensorTree(
        leaf_phys_dims=[2] * n_features,
        ranks=rank,
        out_dim=out_dim,
        seed=seed,
        device=DEVICE,
        dtype=torch.float64,
    ).to(DEVICE)

    # Overwrite cores with our delta-like design to avoid vanishing/exploding
    with torch.no_grad():
        for name, p in teacher.cores.items():
            shape = tuple(p.shape)
            p.copy_(_delta_like_core(shape, g, scale=1.0))

    return teacher


def make_student_tree_like(teacher: BinaryTensorTree, noise_std: float = 1e-2) -> BinaryTensorTree:
    # Same topology/specs as teacher
    student = BinaryTensorTree(
        leaf_phys_dims=[teacher.phys_dims[lid] for lid in teacher.leaf_ids],
        ranks=teacher.ranks,
        out_dim=teacher.out_dim,
        seed=SEED + 1,
        device=DEVICE,
        dtype=torch.float64,
    ).to(DEVICE)

    # Initialize near the teacher to ensure easy/fast convergence
    with torch.no_grad():
        for n in teacher.cores.keys():
            t = teacher.cores[n]
            s = student.cores[n]
            # s.copy_(t + noise_std * torch.randn_like(t, generator=g))
            s.copy_(t + noise_std * torch.randn(t.shape, dtype=t.dtype, device=t.device, generator=g))

    return student


def make_data(n_samples: int, n_features: int) -> torch.Tensor:
    # Features in [0,1], reasonably conditioned. Wider spread can hurt if cores amplify.
    X = torch.rand((n_samples, n_features), dtype=torch.float64, device=DEVICE, generator=g)
    return X


# ---------- Training ----------
def train_student(teacher: BinaryTensorTree, student: BinaryTensorTree, X: torch.Tensor,
                  epochs: int = EPOCHS, batch: int = BATCH, lr: float = LR, clip: float = CLIP) -> Tuple[float, float]:
    teacher.eval()
    with torch.no_grad():
        y = teacher(X).reshape(-1, 1)  # [N,1]

    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    opt = optim.Adam(student.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=False)
    loss_fn = nn.MSELoss()

    best = math.inf
    for epoch in range(1, epochs + 1):
        student.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = student(xb).reshape(-1, 1)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=clip)
            opt.step()
            running += loss.item() * xb.size(0)

        train_mse = running / len(ds)
        sched.step(train_mse)

        if train_mse < best:
            best = train_mse

        if epoch % 25 == 0 or epoch == 1:
            with torch.no_grad():
                pred_full = student(X).reshape(-1, 1)
                full_mse = loss_fn(pred_full, y).item()
                r2 = r2_score(y.flatten(), pred_full.detach())
                print(f"R^2 on train: {r2:.6f}")
            print(f"Epoch {epoch:4d} | batch-MSE={train_mse:.3e} | full-MSE={full_mse:.3e} | lr={opt.param_groups[0]['lr']:.2e}")

        # Early stop if effectively zero
        if best < 1e-14:
            print(f"Early stop at epoch {epoch}: MSE ~ 0")
            break

    with torch.no_grad():
        final_pred = student(X).reshape(-1, 1)
        final_mse = loss_fn(final_pred, y).item()
        rel_err = (torch.norm(final_pred - y) / torch.norm(y)).item()

    return final_mse, rel_err


def main():
    print("Device:", DEVICE)
    teacher = make_teacher_tree(N_FEATURES, RANK, OUT_DIM, seed=SEED)
    student = make_student_tree_like(teacher, noise_std=1e-2)

    X = make_data(N_TRAIN, N_FEATURES)

    final_mse, rel_err = train_student(teacher, student, X, epochs=EPOCHS, batch=BATCH, lr=LR, clip=CLIP)
    print(f"\nFinal train MSE: {final_mse:.3e}")
    print(f"Relative L2 error on train: {rel_err:.3e}")

    # Sanity: cosine similarity on a random probe set
    with torch.no_grad():
        Xp = make_data(4096, N_FEATURES)
        yt = teacher(Xp).flatten()
        ys = student(Xp).flatten()
        cos = (yt @ ys) / (yt.norm() * ys.norm() + 1e-18)
        print(f"Cosine(student, teacher) on probe = {float(cos):.6f}")


if __name__ == "__main__":
    main()
