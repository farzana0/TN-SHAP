#!/usr/bin/env python3
"""
Quick test of the evaluation script with minimal Shapley computation
"""

import sys
sys.path.append('./gpt_explain')

from eval_tn_shapley_orders import load_dataset_and_model
import numpy as np
import torch

def test_model_loading():
    result_path = './tn_results_enhanced/The_food_was_cheap,_fresh,_and_tasty_tn_shapley_enhanced.json'
    
    print("Loading dataset and model...")
    X, y, tokens, model, training_results = load_dataset_and_model(result_path)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Tokens: {tokens}")
    
    # Test model prediction
    print(f"\nTesting model prediction...")
    
    # Prepare sample with bias
    sample_idx = 0
    x_sample = X[sample_idx]  # Shape: [n_tokens, embed_dim]
    
    # Add bias and flatten
    bias_dim = np.ones((1, len(tokens), 1))
    x_sample_with_bias = np.concatenate([x_sample.reshape(1, len(tokens), -1), bias_dim], axis=2)
    x_flat = x_sample_with_bias.flatten()
    
    print(f"Sample shape: {x_flat.shape}")
    print(f"Target value: {y[sample_idx]:.4f}")
    
    # Test prediction function
    def predict_fn(X_input):
        """Prediction function for Shapley computation"""
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)
        X_tensor = torch.FloatTensor(X_input)
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).squeeze().numpy()
            # Ensure we always return an array, even for single predictions
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            return predictions
    
    # Test single prediction
    pred_single = predict_fn(x_flat)
    print(f"Single prediction: {pred_single}, shape: {pred_single.shape}")
    
    # Test batch prediction  
    x_batch = np.stack([x_flat, x_flat])  # 2 identical samples
    pred_batch = predict_fn(x_batch)
    print(f"Batch prediction: {pred_batch}, shape: {pred_batch.shape}")
    
    # Test zero baseline
    x_zero = np.zeros_like(x_flat)
    # But keep bias terms as 1
    for i in range(len(tokens)):
        bias_idx = (i + 1) * (16 + 1) - 1  # Last index for each token
        if bias_idx < len(x_zero):
            x_zero[bias_idx] = 1.0
    
    pred_zero = predict_fn(x_zero)
    print(f"Zero baseline prediction: {pred_zero}, shape: {pred_zero.shape}")
    
    print("\nModel loading and prediction test successful!")
    return True

if __name__ == "__main__":
    test_model_loading()
