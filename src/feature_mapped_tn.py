"""
Feature-mapped tensor networks for TNShap.

This module provides feature mapping capabilities that transform input features
before feeding them to tensor networks, enabling more expressive surrogate models.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Sequence, Union, Tuple, List


# ---- Elementwise feature map that outputs m channels per scalar input ---- #

class ElementwiseFeatureMap(nn.Module):
    """
    Elementwise feature map ψ: R -> R^m applied across features.
    
    Ensures ψ(0) = 0 by using bias=False and activations with a(0)=0.
    This property is important for Shapley value computation.

    Input:  x ∈ R^{B×D}
    Output: z ∈ R^{B×D×m}
    """
    def __init__(self, out_dim: int, hidden: int = 32, act: str = "tanh"):
        """
        Args:
            out_dim: Number of output channels per feature (m)
            hidden: Hidden layer size
            act: Activation function ("tanh", "relu", "gelu", "silu")
        """
        super().__init__()
        assert out_dim >= 1, "out_dim must be >= 1"
        act = act.lower()
        act_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }[act]
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(1, hidden, bias=False),
            act_fn(),
            nn.Linear(hidden, self.out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x: [B, D] -> [B, D, m]
        """
        if x.ndim != 2:
            raise ValueError("ElementwiseFeatureMap expects input of shape [B, D]")
        B, D = x.shape
        z = self.net(x.reshape(B * D, 1)).reshape(B, D, self.out_dim)
        return z


# ---- Selector utilities (thin diagonal in feature-map space) ---- #

SelectorMode = Literal["none", "per_feature_scalar", "per_channel"]

def apply_selectors(
    z: torch.Tensor,                      # [B, F, m]
    t: Optional[torch.Tensor],            # None or shape compatible per mode
    mode: SelectorMode = "none",
) -> torch.Tensor:
    """
    Applies selectors in feature-map space (before bias append).
    
    Args:
        z: Feature-mapped input [B, F, m]
        t: Selector tensor (shape depends on mode)
        mode: Selector mode
            - 'none': no-op
            - 'per_feature_scalar': t ∈ R^{B×F} or {1×F} or {B×1} -> scales all m channels equally per feature
            - 'per_channel': t ∈ R^{B×F×m} (or broadcastable)
    
    Returns:
        Modified feature-mapped input
    """
    if mode == "none" or t is None:
        return z
    if mode == "per_feature_scalar":
        if t.ndim == 1:
            # [F] -> [1,F,1]
            t = t.view(1, -1, 1)
        elif t.ndim == 2:
            # [B,F] or [1,F] or [B,1] -> [B,F,1]
            t = t.unsqueeze(-1)
        elif t.ndim == 3 and t.shape[-1] == 1:
            pass
        else:
            raise ValueError("per_feature_scalar expects t with shape [F], [B,F], [1,F], [B,1] or [B,F,1]")
        return z * t
    if mode == "per_channel":
        # allow broadcasting: [B,F,m] (or [1,F,m], [B,1,m], [1,1,m], etc.)
        return z * t
    raise ValueError(f"Unknown selector mode: {mode}")


# ---- Feature-mapped TN wrapper ------------------------------------- #

class FeatureMappedTN(nn.Module):
    """
    Wraps a BinaryTensorTree with elementwise feature mapping.
    
    Forward path:
        x:[B,F] --ψ--> z:[B,F,m] --selectors--> z':[B,F,m] --append 1--> Xleaves
        Xleaves = list of [B, m+1] (one per leaf), fed to TN.

    You can ablate 'fmap_out_dim' (m) and toggle selector modes.
    """
    def __init__(
        self,
        tn: nn.Module,                     # BinaryTensorTree instance
        d_in: int,                         # number of features F
        fmap_out_dim: int = 1,             # m ≥ 1
        fmap_hidden: int = 32,
        fmap_act: str = "relu",
        selector_mode: SelectorMode = "none",
    ):
        """
        Args:
            tn: BinaryTensorTree instance
            d_in: Number of input features
            fmap_out_dim: Number of feature map output channels
            fmap_hidden: Hidden layer size in feature map
            fmap_act: Activation function for feature map
            selector_mode: Selector mode for ablation studies
        """
        super().__init__()
        self.d_in = int(d_in)
        self.m = int(fmap_out_dim)
        self.selector_mode = selector_mode

        # sanity: TN leaves must have phys_dim = m+1 for every leaf
        # (we append the bias channel here)
        if len(getattr(tn, "leaf_ids")) != self.d_in:
            raise ValueError("TN leaf count must equal d_in")
        leaf_dims = [tn.phys_dims[lid] for lid in tn.leaf_ids]
        if not all(d == self.m + 1 for d in leaf_dims):
            raise ValueError(
                f"TN leaf phys_dims must all be m+1 (= {self.m+1}), but got {leaf_dims}"
            )

        self.feature_map = ElementwiseFeatureMap(out_dim=self.m, hidden=fmap_hidden, act=fmap_act)
        self.tn = tn

    @torch.no_grad()
    def freeze_feature_map(self, freeze: bool = True):
        """Freeze or unfreeze feature map parameters."""
        for p in self.feature_map.parameters():
            p.requires_grad = (not freeze)

    def _build_leaf_inputs(self, z_prime: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert feature-mapped input to per-leaf inputs for TN.
        
        Args:
            z_prime: [B, F, m] feature-mapped input
            
        Returns:
            List of F tensors, each [B, m+1] = [channels, bias]
        """
        B, F, m = z_prime.shape
        device = z_prime.device
        dtype = z_prime.dtype
        ones = torch.ones(B, 1, device=device, dtype=dtype)
        leaf_inputs: List[torch.Tensor] = []
        for j in range(F):
            # concatenate bias as last channel (thin diagonal selectors should leave this at 1)
            leaf_inputs.append(torch.cat([z_prime[:, j, :], ones], dim=1))  # [B, m+1]
        return leaf_inputs

    def forward(
        self,
        x: torch.Tensor,                          # [B, F]
        selectors: Optional[torch.Tensor] = None  # shape depends on selector_mode
    ) -> torch.Tensor:
        """
        Forward pass through feature-mapped tensor network.
        
        Args:
            x: Input features [B, F]
            selectors: Optional selector tensor for ablation studies
            
        Returns:
            Output predictions
        """
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise ValueError("FeatureMappedTN expects x of shape [B, F]")

        # 1) feature map
        z = self.feature_map(x)  # [B, F, m] with ψ(0)=0

        # 2) selectors in feature-map space (bias not present yet, so we don't need to protect it)
        z_prime = apply_selectors(z, selectors, mode=self.selector_mode)  # [B, F, m]

        # 3) append bias channel and feed to TN as per-leaf matrices
        leaf_inputs = self._build_leaf_inputs(z_prime)  # list of F tensors [B, m+1]
        y = self.tn(leaf_inputs)
        return y


# ---- Convenience builders for experiments --------------------------- #

def make_feature_mapped_tn(
    d_in: int,
    fmap_out_dim: int,
    ranks: Union[int, dict],
    out_dim: int = 1,
    fmap_hidden: int = 32,
    fmap_act: str = "relu",
    selector_mode: SelectorMode = "none",
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Builds a BinaryTensorTree with per-leaf phys_dim = fmap_out_dim + 1 (bias),
    and wraps it with FeatureMappedTN.
    
    Args:
        d_in: Number of input features
        fmap_out_dim: Number of feature map output channels
        ranks: Tensor network ranks
        out_dim: Output dimension
        fmap_hidden: Hidden layer size in feature map
        fmap_act: Activation function for feature map
        selector_mode: Selector mode for ablation studies
        seed: Random seed
        device: Device to place tensors on
        dtype: Data type for tensors
        
    Returns:
        FeatureMappedTN instance
    """
    from .tntree_model import make_balanced_binary_tensor_tree  # local import to avoid cycles

    leaf_phys_dims = [fmap_out_dim + 1] * d_in
    tn = make_balanced_binary_tensor_tree(
        leaf_phys_dims=leaf_phys_dims,
        ranks=ranks,
        out_dim=out_dim,
        assume_bias_when_matrix=False,  # we provide [channels, 1] explicitly
        seed=seed,
        device=device,
        dtype=dtype,
    )
    return FeatureMappedTN(
        tn=tn,
        d_in=d_in,
        fmap_out_dim=fmap_out_dim,
        fmap_hidden=fmap_hidden,
        fmap_act=fmap_act,
        selector_mode=selector_mode,
    )


__all__ = [
    "ElementwiseFeatureMap",
    "FeatureMappedTN", 
    "SelectorMode",
    "apply_selectors",
    "make_feature_mapped_tn",
]
