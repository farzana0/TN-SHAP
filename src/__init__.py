"""
TNShap: Tensor Network Shapley Value Estimation

A method for computing Shapley values and higher-order feature interactions 
using Tensor Networks as surrogate models.
"""

from .tntree_model import BinaryTensorTree, make_balanced_binary_tensor_tree
from .feature_mapped_tn import FeatureMappedTN, ElementwiseFeatureMap, make_feature_mapped_tn

__version__ = "1.0.0"
__all__ = [
    "BinaryTensorTree",
    "make_balanced_binary_tensor_tree", 
    "FeatureMappedTN",
    "ElementwiseFeatureMap",
    "make_feature_mapped_tn",
]
