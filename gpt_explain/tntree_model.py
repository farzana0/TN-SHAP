#!/usr/bin/env python3
# tntree_model.py
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Iterable, Union
import torch
import torch.nn as nn


# ------------------------------ Topology ------------------------------ #

def build_balanced_binary_topology(num_leaves: int, node_id_start: int = 0):
    if num_leaves <= 0:
        raise ValueError("num_leaves must be >= 1")
    next_id = node_id_start
    children_map, leaf_ids = {}, []

    def build_range(n_leaves: int) -> int:
        nonlocal next_id
        node = next_id
        next_id += 1
        if n_leaves == 1:
            children_map[node] = (None, None)
            leaf_ids.append(node)
            return node
        n_left = (n_leaves + 1) // 2
        n_right = n_leaves - n_left
        L = build_range(n_left)
        R = build_range(n_right)
        children_map[node] = (L, R)
        return node

    root = build_range(num_leaves)
    return children_map, leaf_ids, root


# ------------------------- Rank specification ------------------------- #
def _expand_ranks(children_map, ranks) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    if isinstance(ranks, int):
        if ranks <= 0:
            raise ValueError("rank must be positive")
        for p, (L, R) in children_map.items():
            if L is not None:
                out[(p, L)] = ranks
            if R is not None:
                out[(p, R)] = ranks
        return out

    if isinstance(ranks, dict):
        # per-edge?
        if all(isinstance(k, tuple) and len(k) == 2 for k in ranks.keys()):
            for p, (L, R) in children_map.items():
                for c in (L, R):
                    if c is None:
                        continue
                    r = int(ranks.get((p, c), 0))
                    if r <= 0:
                        raise ValueError(f"missing/invalid rank for edge {(p, c)}")
                    out[(p, c)] = r
            return out
        # per-node {node: (rL, rR)}
        for p, (L, R) in children_map.items():
            if L is None:
                continue
            if p not in ranks:
                raise ValueError(f"missing ranks for node {p}")
            rL, rR = map(int, ranks[p])
            if rL <= 0 or rR <= 0:
                raise ValueError(f"invalid node ranks at {p}")
            out[(p, L)] = rL
            out[(p, R)] = rR
            continue

    raise TypeError("ranks must be int, {(p,c): r}, or {node: (rL,rR)}")


# --------------------------- BinaryTensorTree -------------------------- #

class BinaryTensorTree(nn.Module):
    def __init__(
        self,
        leaf_phys_dims: Iterable[int],
        ranks: Union[int, Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]]],
        out_dim: int = 1,
        assume_bias_when_matrix: bool = True,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        *,
        leaf_input_dims: Union[int, Iterable[int]] = 1,
    ):
        """
        Args:
            leaf_phys_dims: physical dimension per leaf (e.g., d_i or d_i+1 if including bias channel).
            leaf_input_dims: feature dimension per leaf (d_i). If an int, broadcast to all leaves.
                             If iterable, must match number of leaves.
        """
        super().__init__()
        dims = list(leaf_phys_dims)
        children_map, leaf_ids, root = build_balanced_binary_topology(len(dims))
        self.children_map, self.leaf_ids, self.root = children_map, leaf_ids, root
        self.phys_dims = {lid: d for lid, d in zip(leaf_ids, dims)}
        self.out_dim = int(out_dim)
        self.assume_bias_when_matrix = bool(assume_bias_when_matrix)
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Input feature dims per leaf (d_i)
        if isinstance(leaf_input_dims, int):
            in_dims_list = [int(leaf_input_dims)] * len(self.leaf_ids)
        else:
            in_dims_list = list(leaf_input_dims)
            if len(in_dims_list) != len(self.leaf_ids):
                raise ValueError("leaf_input_dims length must match number of leaves")
            in_dims_list = [int(d) for d in in_dims_list]
        if any(d <= 0 for d in in_dims_list):
            raise ValueError("leaf_input_dims must be positive integers")
        self.in_dims: Dict[int, int] = {lid: d for lid, d in zip(self.leaf_ids, in_dims_list)}

        # Validate phys vs input dims (allow bias if requested)
        for lid in self.leaf_ids:
            phys = self.phys_dims[lid]
            din = self.in_dims[lid]
            if phys not in (din, din + 1):
                raise ValueError(
                    f"Leaf {lid}: phys_dim={phys} must equal d={din} or d+1 (for bias)."
                )
            if phys == din + 1 and not self.assume_bias_when_matrix:
                # We can still accept list inputs that already include bias, but warn for matrices.
                pass

        # Precompute column slices if we receive a big [B, sum(d_i)] matrix
        self._col_slices: Dict[int, Tuple[int, int]] = {}
        start = 0
        for lid in self.leaf_ids:
            din = self.in_dims[lid]
            self._col_slices[lid] = (start, start + din)
            start += din
        self._total_in_features = start  # sum(d_i)

        # ranks
        self.ranks = _expand_ranks(self.children_map, ranks)

        # parameters
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        self.cores = nn.ParameterDict()
        self.leaf_mlps = nn.ModuleDict()  # per-leaf feature maps: d_i -> (phys_dim_i * rin)

        for n in self._preorder(self.root):
            p = self._parent(n)
            L, R = self.children_map[n]
            if L is None and R is None:
                # Leaf: dynamic core via tiny MLP (Linear(d_i -> phys_dim * rin))
                rin = self.ranks[(p, n)] if p is not None else (self.out_dim if self.out_dim > 1 else 1)
                phys = self.phys_dims[n]
                din = self.in_dims[n]
                mlp = nn.Sequential(nn.Linear(din, phys * rin, bias=True))
                mlp.to(device=self.device, dtype=self.dtype)
                self.leaf_mlps[str(n)] = mlp
            elif n == self.root:
                rl = self.ranks[(n, L)]
                rr = self.ranks[(n, R)]
                shape = (rl, rr) if self.out_dim == 1 else (rl, rr, self.out_dim)
                self.cores[str(n)] = nn.Parameter(self._init_tensor(shape, g))
            else:
                rl = self.ranks[(n, L)]
                rr = self.ranks[(n, R)]
                rin = self.ranks[(p, n)]
                shape = (rl, rr, rin)
                self.cores[str(n)] = nn.Parameter(self._init_tensor(shape, g))

        # build einsum eqs (kept for compatibility; forward() uses hierarchical path)
        self._eq_batch, self._eq_nobatch = self._build_equations()

    # ------------ helpers ------------
    def _parent(self, node: int) -> Optional[int]:
        for p, (L, R) in self.children_map.items():
            if L == node or R == node:
                return p
        return None

    def _preorder(self, node: int) -> List[int]:
        out = []

        def visit(n):
            out.append(n)
            L, R = self.children_map[n]
            if L is not None:
                visit(L)
            if R is not None:
                visit(R)

        visit(node)
        return out

    def _init_tensor(self, shape, gen):
        t = torch.empty(*shape, device=self.device, dtype=self.dtype)
        fan_in = math.prod(shape[:-1]) if len(shape) > 1 else 1
        fan_out = shape[-1] if len(shape) > 0 else 1
        std = math.sqrt(2.0 / (fan_in + fan_out))
        try:
            torch.nn.init.normal_(t, mean=0.0, std=std, generator=gen)
        except TypeError:
            torch.nn.init.normal_(t, mean=0.0, std=std)
        return t

    def _build_equations(self):
        # label axes
        next_ax = 0
        phys_id = {}
        for lid in self.leaf_ids:
            phys_id[lid] = next_ax
            next_ax += 1
        bond_id = {}
        for p, (L, R) in self.children_map.items():
            for c in (L, R):
                if c is not None:
                    bond_id[(p, c)] = next_ax
                    next_ax += 1
        out_axis = None
        if self.out_dim > 1:
            out_axis = next_ax
            next_ax += 1

        def id2sym(i):
            if i < 26:
                return chr(ord("a") + i)
            if i < 52:
                return chr(ord("A") + (i - 26))
            raise ValueError("too many indices for einsum")

        def term(ids):
            return "".join(id2sym(i) for i in ids)

        node_axes = {}
        for n in self._preorder(self.root):
            p = self._parent(n)
            L, R = self.children_map[n]
            if L is None and R is None:
                axes = [phys_id[n]]
                if p is not None:
                    axes.append(bond_id[(p, n)])
                elif out_axis is not None:
                    axes.append(out_axis)
            elif n == self.root:
                axes = [bond_id[(n, L)], bond_id[(n, R)]]
                if out_axis is not None:
                    axes.append(out_axis)
            else:
                axes = [bond_id[(n, L)], bond_id[(n, R)], bond_id[(self._parent(n), n)]]
            node_axes[n] = axes

        # batch axis
        all_ids = set()
        for ax in node_axes.values():
            all_ids.update(ax)
        BAXIS = (max(all_ids) + 1) if all_ids else 0

        # equation strings
        ops_terms = ["".join(term(node_axes[n])) for n in self._preorder(self.root)]

        leaves_batch = ["".join(term([BAXIS, node_axes[lid][0]])) for lid in self.leaf_ids]
        leaves_nobatch = ["".join(term([node_axes[lid][0]])) for lid in self.leaf_ids]

        out_batch = "".join([term([BAXIS]) + (term([node_axes[self.root][-1]]) if self.out_dim > 1 else "")])
        out_nobatch = "".join([(term([node_axes[self.root][-1]]) if self.out_dim > 1 else "")])

        eq_batch = ",".join(ops_terms + leaves_batch) + "->" + out_batch
        eq_nobatch = ",".join(ops_terms + leaves_nobatch) + "->" + out_nobatch
        return eq_batch, eq_nobatch

    # ------------ input conversion ------------
    def _matrix_to_inputs(self, X: torch.Tensor):
        """
        Accept a big matrix X of shape [B, sum(d_i)] or [sum(d_i)] and split it per leaf.
        Produce per-leaf physical vectors of shape:
            - [B, phys_dim_i] if batched
            - [phys_dim_i] if single example
        If phys_dim_i == d_i + 1 and assume_bias_when_matrix=True, append a bias-1 channel.
        """
        if X.ndim not in (1, 2):
            raise ValueError("X must be [F_total] or [B, F_total]")
        batched = (X.ndim == 2)
        if batched:
            B, F = X.shape
            if F != self._total_in_features:
                raise ValueError(f"feature mismatch: got {F}, expected {self._total_in_features}")
        else:
            F = X.shape[0]
            if F != self._total_in_features:
                raise ValueError(f"feature mismatch: got {F}, expected {self._total_in_features}")

        xs = []
        for lid in self.leaf_ids:
            s, e = self._col_slices[lid]
            din = self.in_dims[lid]
            phys = self.phys_dims[lid]
            if batched:
                feat = X[:, s:e]  # [B, d_i]
                if phys == din:
                    xs.append(feat)
                elif phys == din + 1 and self.assume_bias_when_matrix:
                    ones = torch.ones(feat.size(0), 1, dtype=feat.dtype, device=feat.device)
                    xs.append(torch.cat([feat, ones], dim=1))  # [B, d_i+1]
                else:
                    raise ValueError(f"Leaf {lid}: cannot realize phys_dim={phys} from d={din} "
                                     f"(assume_bias_when_matrix={self.assume_bias_when_matrix}).")
            else:
                feat = X[s:e]  # [d_i]
                if phys == din:
                    xs.append(feat)
                elif phys == din + 1 and self.assume_bias_when_matrix:
                    one = torch.ones(1, dtype=feat.dtype, device=feat.device)
                    xs.append(torch.cat([feat, one], dim=0))  # [d_i+1]
                else:
                    raise ValueError(f"Leaf {lid}: cannot realize phys_dim={phys} from d={din} "
                                     f"(assume_bias_when_matrix={self.assume_bias_when_matrix}).")
        return xs, batched

    # ------------ forward ------------
    def forward_einsum(self, inputs):
        # NOTE: kept for API parity, but does NOT use the MLP leaf maps.
        # forward() calls forward_hierarchical(), which DOES use the MLPs.
        if isinstance(inputs, torch.Tensor):
            inputs_by_leaf, batched = self._matrix_to_inputs(inputs)
        else:
            inputs_by_leaf = inputs
        ops = [self.cores[str(n)] for n in self._preorder(self.root) if n != self.root or True] + inputs_by_leaf
        eq = self._eq_batch if (isinstance(inputs_by_leaf[0], torch.Tensor) and inputs_by_leaf[0].ndim == 2) else self._eq_nobatch
        return torch.einsum(eq, *ops)

    def forward_hierarchical(self, inputs):
        # Accept either:
        #   - Tensor: [F_total] or [B, F_total], where F_total = sum(d_i)
        #   - List of per-leaf tensors with shapes:
        #       [B, phys_i] or [phys_i]  (already-constructed physical vectors)
        if isinstance(inputs, torch.Tensor):
            X = inputs if inputs.ndim == 2 else inputs.unsqueeze(0)  # [B, F_total]
            single = (inputs.ndim == 1)
            xs_phys, batched = self._matrix_to_inputs(X if not single else inputs)
        else:
            xs_phys = inputs
            batched = (xs_phys[0].ndim == 2)
            X = None
            single = False

        node_vals = {}
        # leaves
        for i, lid in enumerate(self.leaf_ids):
            x_phys = xs_phys[i]  # [B, phys] or [phys]
            phys = self.phys_dims[lid]
            din = self.in_dims[lid]

            # Prepare MLP input features x_feat (size d_i)
            if X is not None:
                # Slice original matrix
                s, e = self._col_slices[lid]
                x_feat = X[:, s:e]  # [B, d_i]
            else:
                # Recover from provided physical vector (take first d_i channels)
                if batched:
                    x_feat = x_phys[:, :din]
                else:
                    x_feat = x_phys[:din].unsqueeze(0)  # [1, d_i]

            rin = self.ranks[(self._parent(lid), lid)] if self._parent(lid) is not None else (self.out_dim if self.out_dim > 1 else 1)
            mlp = self.leaf_mlps[str(lid)]
            core_flat = mlp(x_feat.to(dtype=self.dtype))  # [B, phys*rin]
            if batched:
                core_dyn = core_flat.view(-1, phys, rin)   # [B, phys, rin]
                # contraction over physical dimension -> [B, rin]
                up = (x_phys.to(dtype=self.dtype).unsqueeze(-1) * core_dyn).sum(dim=1)
            else:
                core_dyn = core_flat.view(1, phys, rin)[0]  # [phys, rin]
                up = (x_phys.to(dtype=self.dtype).unsqueeze(-1) * core_dyn).sum(dim=0)  # [rin]
            node_vals[lid] = up

        # internal up to root (unchanged)
        for n in reversed(self._preorder(self.root)):
            L, R = self.children_map[n]
            if L is None and R is None:
                continue
            core = self.cores[str(n)]
            left = node_vals[L]
            right = node_vals[R]
            if n == self.root:
                if self.out_dim == 1:
                    if batched:
                        tmp = torch.tensordot(left, core, dims=([1], [0]))
                        y = (tmp * right).sum(dim=1)
                    else:
                        tmp = torch.tensordot(left, core, dims=([0], [0]))
                        y = (tmp * right).sum(dim=0)
                    return y if not single else y.squeeze(0)
                else:
                    if batched:
                        tmp = torch.tensordot(left, core, dims=([1], [0]))
                        out = (right.unsqueeze(-1) * tmp).sum(dim=1)
                    else:
                        tmp = torch.tensordot(left, core, dims=([0], [0]))
                        out = (right.unsqueeze(-1) * tmp).sum(dim=0)
                    return out if not single else out.squeeze(0)
            else:
                if batched:
                    tmp = torch.tensordot(left, core, dims=([1], [0]))
                    up = (right.unsqueeze(-1) * tmp).sum(dim=1)
                else:
                    tmp = torch.tensordot(left, core, dims=([0], [0]))
                    up = (right.unsqueeze(-1) * tmp).sum(dim=0)
                node_vals[n] = up

    def forward(self, inputs):
        # Keep using the hierarchical path so the new MLPs are part of the computation graph.
        return self.forward_hierarchical(inputs)

    # expose scikit-like .predict
    def predict(self, inputs):
        return self.forward(inputs)


# -------- convenience builder to match your older imports -------- #

def make_balanced_binary_tensor_tree(
    leaf_phys_dims: Iterable[int],
    ranks: Union[int, Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]]],
    out_dim: int = 1,
    assume_bias_when_matrix: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    *,
    leaf_input_dims: Union[int, Iterable[int]] = 1,
) -> BinaryTensorTree:
    return BinaryTensorTree(
        leaf_phys_dims=leaf_phys_dims,
        ranks=ranks,
        out_dim=out_dim,
        assume_bias_when_matrix=assume_bias_when_matrix,
        seed=seed,
        device=device,
        dtype=dtype,
        leaf_input_dims=leaf_input_dims,
    )


__all__ = [
    "BinaryTensorTree",
    "build_balanced_binary_topology",
    "make_balanced_binary_tensor_tree",
]
