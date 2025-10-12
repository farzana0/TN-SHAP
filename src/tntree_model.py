#!/usr/bin/env python3
"""
Binary Tensor Tree implementation for TNShap.

This module provides a balanced binary tensor tree architecture that can be used
as a surrogate model for computing Shapley values and higher-order feature interactions.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Iterable, Union
import torch
import torch.nn as nn


# ------------------------------ Topology ------------------------------ #

def build_balanced_binary_topology(num_leaves: int, node_id_start: int = 0):
    """
    Build a balanced binary tree topology for tensor networks.
    
    Args:
        num_leaves: Number of leaf nodes (features)
        node_id_start: Starting ID for node numbering
        
    Returns:
        children_map: Dict mapping parent nodes to (left_child, right_child)
        leaf_ids: List of leaf node IDs
        root: Root node ID
    """
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
# ranks can be:
#   - int: same rank on every edge
#   - dict[(parent, child)] = rank: per-edge ranks
#   - dict[parent] = (rank_left, rank_right): per-node ranks
def _expand_ranks(children_map, ranks) -> Dict[Tuple[int, int], int]:
    """Expand rank specification into per-edge ranks."""
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
    """
    Balanced binary tensor tree for efficient computation of multilinear functions.
    
    This implementation supports both einsum-based and hierarchical forward passes,
    automatically falling back to hierarchical computation for large dimensions
    where einsum would exceed the 52-symbol limit.
    
    Args:
        leaf_phys_dims: Physical dimensions for each leaf node
        ranks: Tensor network ranks (int, dict, or per-node specification)
        out_dim: Output dimension (default: 1)
        assume_bias_when_matrix: Whether to append bias when processing matrix inputs
        seed: Random seed for initialization
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    def __init__(
        self,
        leaf_phys_dims: Iterable[int],
        ranks: Union[int, Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]]],
        out_dim: int = 1,
        assume_bias_when_matrix: bool = True,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        dims = list(leaf_phys_dims)
        children_map, leaf_ids, root = build_balanced_binary_topology(len(dims))
        self.children_map, self.leaf_ids, self.root = children_map, leaf_ids, root
        self.phys_dims = {lid: d for lid, d in zip(leaf_ids, dims)}
        self.out_dim = int(out_dim)
        self.assume_bias_when_matrix = bool(assume_bias_when_matrix)
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # ranks
        self.ranks = _expand_ranks(self.children_map, ranks)

        # parameters
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        self.cores = nn.ParameterDict()
        for n in self._preorder(self.root):
            p = self._parent(n)
            L, R = self.children_map[n]
            if L is None and R is None:
                rin = self.ranks[(p, n)] if p is not None else (self.out_dim if self.out_dim > 1 else 1)
                shape = (self.phys_dims[n], rin)
            elif n == self.root:
                rl = self.ranks[(n, L)]
                rr = self.ranks[(n, R)]
                shape = (rl, rr) if self.out_dim == 1 else (rl, rr, self.out_dim)
            else:
                rl = self.ranks[(n, L)]
                rr = self.ranks[(n, R)]
                rin = self.ranks[(p, n)]
                shape = (rl, rr, rin)
            self.cores[str(n)] = nn.Parameter(self._init_tensor(shape, g))

        # Try to build einsum equations; if too many indices, skip and use hierarchical.
        self._eq_batch, self._eq_nobatch = None, None
        try:
            self._eq_batch, self._eq_nobatch = self._build_equations()
        except ValueError:
            # Too many indices for torch.einsum (limit is 52: a..zA..Z) — fall back silently.
            self._eq_batch, self._eq_nobatch = None, None

    # ------------ helpers ------------
    def _parent(self, node: int) -> Optional[int]:
        """Find parent of a node."""
        for p, (L, R) in self.children_map.items():
            if L == node or R == node:
                return p
        return None

    def _preorder(self, node: int) -> List[int]:
        """Get nodes in preorder traversal."""
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
        """Initialize tensor with Xavier-like initialization."""
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
        """Build einsum equations for forward pass."""
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

        # We will also add one batch axis later.
        # torch.einsum only allows 52 unique symbols (a..zA..Z).
        TOTAL_AXES = next_ax + 1  # +1 for batch axis
        if TOTAL_AXES > 52:
            # Signal caller to skip einsum and use hierarchical forward.
            raise ValueError("too many indices for einsum")

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
        """Convert matrix input to per-leaf inputs."""
        if X.ndim != 2:
            raise ValueError("X must be [B,F]")
        B, F = X.shape
        if F != len(self.leaf_ids):
            raise ValueError("feature mismatch")
        xs = []
        if self.assume_bias_when_matrix:
            ones = torch.ones(B, 1, dtype=X.dtype, device=X.device)
        for i, lid in enumerate(self.leaf_ids):
            d_i = self.phys_dims[lid]
            col = X[:, i:i+1]
            if d_i == 1:
                xs.append(col)
            elif d_i == 2 and self.assume_bias_when_matrix:
                xs.append(torch.cat([col, ones], dim=1))
            else:
                raise ValueError(f"leaf {lid} expects phys_dim {d_i}")
        return xs

    # ------------ forward ------------
    def forward_einsum(self, inputs):
        """Forward pass using einsum (faster for small dimensions)."""
        # If einsum equations are unavailable (too many indices), force caller to use hierarchical.
        if self._eq_batch is None or self._eq_nobatch is None:
            raise RuntimeError("einsum equation unavailable; use forward_hierarchical instead")

        if isinstance(inputs, torch.Tensor):
            inputs_by_leaf = self._matrix_to_inputs(inputs)
        else:
            inputs_by_leaf = inputs
        batched = (inputs_by_leaf[0].ndim == 2)
        ops = [self.cores[str(n)] for n in self._preorder(self.root)] + inputs_by_leaf
        eq = self._eq_batch if batched else self._eq_nobatch
        return torch.einsum(eq, *ops)

    def forward_hierarchical(self, inputs):
        """Forward pass using hierarchical computation (works for any dimension)."""
        if isinstance(inputs, torch.Tensor):
            xs = self._matrix_to_inputs(inputs)
        else:
            xs = inputs
        batched = (xs[0].ndim == 2)

        node_vals = {}
        # leaves
        for lid, x in zip(self.leaf_ids, xs):
            core = self.cores[str(lid)]
            if batched:
                node_vals[lid] = torch.tensordot(x, core, dims=([1], [0]))
            else:
                node_vals[lid] = torch.tensordot(x, core, dims=([0], [0]))
        # internal up to root
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
                    return y
                else:
                    if batched:
                        tmp = torch.tensordot(left, core, dims=([1], [0]))
                        return (right.unsqueeze(-1) * tmp).sum(dim=1)
                    else:
                        tmp = torch.tensordot(left, core, dims=([0], [0]))
                        return (right.unsqueeze(-1) * tmp).sum(dim=0)
            else:
                if batched:
                    tmp = torch.tensordot(left, core, dims=([1], [0]))
                    up = (right.unsqueeze(-1) * tmp).sum(dim=1)
                else:
                    tmp = torch.tensordot(left, core, dims=([0], [0]))
                    up = (right.unsqueeze(-1) * tmp).sum(dim=0)
                node_vals[n] = up

    def forward(self, inputs):
        """Default forward pass (uses hierarchical computation)."""
        # Default forward uses the robust hierarchical path (works for any d).
        return self.forward_hierarchical(inputs)

    # expose scikit-like .predict
    def predict(self, inputs):
        """Scikit-learn compatible predict method."""
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
) -> BinaryTensorTree:
    """
    Convenience function to create a balanced binary tensor tree.
    
    Args:
        leaf_phys_dims: Physical dimensions for each leaf
        ranks: Tensor network ranks
        out_dim: Output dimension
        assume_bias_when_matrix: Whether to append bias for matrix inputs
        seed: Random seed
        device: Device to place tensors on
        dtype: Data type for tensors
        
    Returns:
        BinaryTensorTree instance
    """
    return BinaryTensorTree(
        leaf_phys_dims=leaf_phys_dims,
        ranks=ranks,
        out_dim=out_dim,
        assume_bias_when_matrix=assume_bias_when_matrix,
        seed=seed,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "BinaryTensorTree",
    "build_balanced_binary_topology",
    "make_balanced_binary_tensor_tree",
]
