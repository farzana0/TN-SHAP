"""
Shapley value computation utilities for TNShap.
"""

import torch
import numpy as np
from typing import Union, Tuple


def compute_shapley_values_tnshap(
    model: torch.nn.Module, 
    x_point: torch.Tensor, 
    order: int = 1,
    baseline: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    Compute Shapley values using TNShap (tensor network surrogate model).
    
    Args:
        model: Trained tensor network model
        x_point: Input point to explain (shape: [1, d])
        order: Order of Shapley values to compute (1 for individual, 2 for pairwise, etc.)
        baseline: Baseline values (default: zeros)
    
    Returns:
        Shapley values (shape: [1, d] for order=1)
    """
    model.eval()
    d = x_point.shape[1]
    
    if baseline is None:
        baseline = torch.zeros_like(x_point)
    
    with torch.no_grad():
        if order == 1:
            return _compute_order1_shapley(model, x_point, baseline)
        elif order == 2:
            return _compute_order2_shapley(model, x_point, baseline)
        else:
            raise ValueError(f"Order {order} not implemented. Use order=1 or order=2.")


def _compute_order1_shapley(
    model: torch.nn.Module, 
    x_point: torch.Tensor, 
    baseline: torch.Tensor
) -> torch.Tensor:
    """Compute first-order (individual) Shapley values."""
    d = x_point.shape[1]
    shapley_values = torch.zeros_like(x_point)
    
    # Baseline prediction
    baseline_pred = model(baseline)
    
    # Individual contributions
    for i in range(d):
        # Create intervention: set feature i to x_point value
        intervention = baseline.clone()
        intervention[0, i] = x_point[0, i]
        
        # Compute prediction with intervention
        intervention_pred = model(intervention)
        
        # Shapley value is the difference
        shapley_values[0, i] = intervention_pred - baseline_pred
    
    return shapley_values


def _compute_order2_shapley(
    model: torch.nn.Module, 
    x_point: torch.Tensor, 
    baseline: torch.Tensor
) -> torch.Tensor:
    """Compute second-order (pairwise interaction) Shapley values."""
    d = x_point.shape[1]
    shapley_interactions = torch.zeros((1, d, d))
    
    # Baseline prediction
    baseline_pred = model(baseline)
    
    # Individual effects (for computing interactions)
    individual_effects = torch.zeros_like(x_point)
    for i in range(d):
        intervention = baseline.clone()
        intervention[0, i] = x_point[0, i]
        individual_effects[0, i] = model(intervention) - baseline_pred
    
    # Pairwise interactions
    for i in range(d):
        for j in range(i + 1, d):
            # Joint intervention
            joint_intervention = baseline.clone()
            joint_intervention[0, i] = x_point[0, i]
            joint_intervention[0, j] = x_point[0, j]
            
            joint_pred = model(joint_intervention)
            joint_effect = joint_pred - baseline_pred
            
            # Interaction is joint effect minus individual effects
            interaction = joint_effect - individual_effects[0, i] - individual_effects[0, j]
            
            shapley_interactions[0, i, j] = interaction
            shapley_interactions[0, j, i] = interaction
    
    return shapley_interactions


def compute_shapley_values_sampling(
    model: torch.nn.Module,
    x_point: torch.Tensor,
    n_samples: int = 1000,
    baseline: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    Compute Shapley values using sampling-based approach (KernelSHAP-like).
    
    Args:
        model: Trained model
        x_point: Input point to explain
        n_samples: Number of samples for Monte Carlo estimation
        baseline: Baseline values
    
    Returns:
        Shapley values
    """
    model.eval()
    d = x_point.shape[1]
    
    if baseline is None:
        baseline = torch.zeros_like(x_point)
    
    shapley_values = torch.zeros_like(x_point)
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Random subset of features
            subset = torch.rand(d) < 0.5
            
            # Create intervention
            intervention = baseline.clone()
            intervention[0, subset] = x_point[0, subset]
            
            # Compute prediction
            pred = model(intervention)
            
            # Update Shapley values (simplified sampling)
            shapley_values[0, subset] += pred / n_samples
    
    return shapley_values
