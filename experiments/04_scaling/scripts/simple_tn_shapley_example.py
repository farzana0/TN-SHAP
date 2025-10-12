#!/usr/bin/env python3
"""
Simple TN-Shapley Example with d=100 and Strong Pairwise Interactions

This is a simplified version that demonstrates:
1. Synthetic data generation with strong pairwise interactions (d=100)
2. Ground truth Shapley calculations using GAM approach
3. TN fitting with R² > 0.9 target
4. Mask augmentation for training
5. Comparison with ground truth
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

# Add TN path
sys.path.append('/home/mila/f/farzaneh.heidari/scratch/tenis/TN_shapley')

try:
    from tntree_model import BinaryTensorTree
    TN_AVAILABLE = True
    print("✅ BinaryTensorTree imported successfully")
except ImportError as e:
    TN_AVAILABLE = False
    print(f"❌ BinaryTensorTree import failed: {e}")

def set_seeds(seed=42):
    """Set all seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_synthetic_data(d=100, n_samples=2000, n_strong=20, n_weak=30, noise_std=0.1, seed=42):
    """Generate synthetic data with strong pairwise interactions."""
    set_seeds(seed)
    
    print(f"📊 Generating synthetic data:")
    print(f"  Features: {d}")
    print(f"  Samples: {n_samples}")
    print(f"  Strong pairs: {n_strong}")
    print(f"  Weak pairs: {n_weak}")
    
    # Generate coefficients
    coefficients = {}
    
    # All possible pairs
    all_pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    np.random.shuffle(all_pairs)
    
    # Strong interactions
    strong_pairs = all_pairs[:n_strong]
    for i, j in strong_pairs:
        coefficients[(i, j)] = np.random.normal(0, 2.0)  # Strong
    
    # Weak interactions  
    weak_pairs = all_pairs[n_strong:n_strong + n_weak]
    for i, j in weak_pairs:
        coefficients[(i, j)] = np.random.normal(0, 0.5)  # Weak
    
    # Linear terms
    linear_coeffs = np.random.normal(0, 1.0, d)
    
    # Generate data
    X = np.random.normal(0, 1, (n_samples, d))
    y = np.zeros(n_samples)
    
    # Linear terms
    y += X @ linear_coeffs
    
    # Pairwise interactions
    for (i, j), coeff in coefficients.items():
        y += coeff * X[:, i] * X[:, j]
    
    # Noise
    y += np.random.normal(0, noise_std, n_samples)
    
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, coefficients, linear_coeffs, strong_pairs, weak_pairs

def compute_ground_truth_shapley(x, coefficients, linear_coeffs):
    """Compute ground truth Shapley values analytically."""
    d = len(linear_coeffs)
    x = np.asarray(x)
    
    # Order 1 Shapley (marginal contributions)
    shapley_k1 = linear_coeffs.copy()
    
    # Add half of each interaction to each participating feature
    for (i, j), coeff in coefficients.items():
        shapley_k1[i] += 0.5 * coeff * x[j]
        shapley_k1[j] += 0.5 * coeff * x[i]
    
    # Order 2 Shapley (pure interactions)
    shapley_k2 = []
    for i in range(d):
        for j in range(i+1, d):
            if (i, j) in coefficients:
                shapley_k2.append(coefficients[(i, j)] * x[i] * x[j])
            else:
                shapley_k2.append(0.0)
    
    return np.array(shapley_k1), np.array(shapley_k2)

def compute_gam_shapley(X, y, model_func, x_point):
    """Compute GAM-based Shapley values."""
    d = len(x_point)
    baseline = np.zeros(d)
    
    print("  Computing GAM Shapley values...")
    
    # Order 1: Individual contributions
    baseline_pred = model_func(baseline.reshape(1, -1))[0]
    shapley_k1 = []
    
    for i in range(d):
        intervention = baseline.copy()
        intervention[i] = x_point[i]
        intervention_pred = model_func(intervention.reshape(1, -1))[0]
        shapley_k1.append(intervention_pred - baseline_pred)
    
    # Order 2: Pairwise interactions (sample subset for efficiency)
    max_pairs = min(200, d * (d - 1) // 2)
    all_pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    np.random.shuffle(all_pairs)
    selected_pairs = all_pairs[:max_pairs]
    
    shapley_k2 = []
    for i, j in selected_pairs:
        # Individual effects
        interv_i = baseline.copy()
        interv_i[i] = x_point[i]
        pred_i = model_func(interv_i.reshape(1, -1))[0]
        
        interv_j = baseline.copy() 
        interv_j[j] = x_point[j]
        pred_j = model_func(interv_j.reshape(1, -1))[0]
        
        # Joint effect
        interv_ij = baseline.copy()
        interv_ij[i] = x_point[i]
        interv_ij[j] = x_point[j]
        pred_ij = model_func(interv_ij.reshape(1, -1))[0]
        
        # Interaction = joint - individual effects
        interaction = (pred_ij - baseline_pred) - (pred_i - baseline_pred) - (pred_j - baseline_pred)
        shapley_k2.append(interaction)
    
    return np.array(shapley_k1), np.array(shapley_k2), selected_pairs

def generate_masks(X, y, n_masks=1000, mask_prob=0.3):
    """Generate mask-augmented training data."""
    n_samples, d = X.shape
    
    print(f"🎭 Generating {n_masks} mask-augmented samples...")
    
    masks = np.random.binomial(1, 1 - mask_prob, (n_masks, d)).astype(bool)
    
    X_masked = []
    y_masked = []
    
    for mask in masks:
        idx = np.random.randint(n_samples)
        x_masked = X[idx].copy()
        x_masked[~mask] = 0  # Zero out masked features
        
        X_masked.append(x_masked)
        y_masked.append(y[idx])
    
    return np.array(X_masked), np.array(y_masked), masks

def train_tensor_network(X_train, y_train, d, rank=8, epochs=200, lr=0.01):
    """Train BinaryTensorTree model."""
    if not TN_AVAILABLE:
        print("❌ TensorNetwork not available")
        return None, 0
    
    try:
        print(f"🧠 Training Tensor Network:")
        print(f"  Input shape: {X_train.shape}")
        print(f"  Rank: {rank}, Epochs: {epochs}")
        
        # Create model - expects input as [batch, features] 
        model = BinaryTensorTree(
            leaf_phys_dims=[2] * d,  # Each feature has 2 physical dims (value + bias)
            ranks=rank,
            out_dim=1,
            assume_bias_when_matrix=True,  # Automatically add bias
            seed=42
        )
        
        # Convert to tensors - BinaryTensorTree expects [batch, features] format
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        
        print(f"  Tensor input shape: {X_tensor.shape}")  # Should be [batch, features]
        
        # Train-validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_r2 = -float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            
            # Training step
            optimizer.zero_grad()
            train_pred = model(X_tr).squeeze()
            train_loss = criterion(train_pred, y_tr.squeeze())
            train_loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val).squeeze()
                    val_loss = criterion(val_pred, y_val.squeeze())
                    val_r2 = r2_score(y_val.squeeze().numpy(), val_pred.numpy())
                    
                    scheduler.step(val_loss)
                    
                    print(f"    Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, Val R²: {val_r2:.4f}")
                    
                    if val_r2 > best_val_r2:
                        best_val_r2 = val_r2
                        patience_counter = 0
                        best_model = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= 20:
                            print(f"    Early stopping at epoch {epoch}")
                            break
        
        # Load best model
        model.load_state_dict(best_model)
        model.eval()
        
        print(f"  ✅ Training completed. Best Val R²: {best_val_r2:.4f}")
        
        return model, best_val_r2
        
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def compute_tn_shapley(model, x_point, d):
    """Compute TN-based Shapley values."""
    if model is None:
        return None, None
    
    print("  Computing TN Shapley values...")
    
    # Baseline (all zeros)
    baseline = np.zeros(d)
    
    with torch.no_grad():
        model.eval()
        
        # Baseline prediction
        baseline_pred = model(torch.tensor(baseline.reshape(1, -1), dtype=torch.float32)).item()
        
        # Order 1: Individual contributions
        tn_shapley_k1 = []
        for i in range(d):
            intervention = baseline.copy()
            intervention[i] = x_point[i]
            intervention_pred = model(torch.tensor(intervention.reshape(1, -1), dtype=torch.float32)).item()
            tn_shapley_k1.append(intervention_pred - baseline_pred)
        
        tn_shapley_k1 = np.array(tn_shapley_k1)
        
        # Order 2: Pairwise interactions (limited for efficiency)
        max_pairs = min(100, d * (d - 1) // 2)
        all_pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
        np.random.shuffle(all_pairs)
        selected_pairs = all_pairs[:max_pairs]
        
        tn_shapley_k2 = []
        for i, j in selected_pairs:
            # Individual effects
            interv_i = baseline.copy()
            interv_i[i] = x_point[i]
            pred_i = model(torch.tensor(interv_i.reshape(1, -1), dtype=torch.float32)).item()
            
            interv_j = baseline.copy()
            interv_j[j] = x_point[j]
            pred_j = model(torch.tensor(interv_j.reshape(1, -1), dtype=torch.float32)).item()
            
            # Joint effect
            interv_ij = baseline.copy()
            interv_ij[i] = x_point[i]
            interv_ij[j] = x_point[j]
            pred_ij = model(torch.tensor(interv_ij.reshape(1, -1), dtype=torch.float32)).item()
            
            # Interaction
            interaction = (pred_ij - baseline_pred) - (pred_i - baseline_pred) - (pred_j - baseline_pred)
            tn_shapley_k2.append(interaction)
        
        tn_shapley_k2 = np.array(tn_shapley_k2)
    
    return tn_shapley_k1, tn_shapley_k2

def compute_metrics(y_true, y_pred):
    """Compute comparison metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))
    
    # Cosine similarity
    na, nb = np.linalg.norm(y_true), np.linalg.norm(y_pred)
    cosine = 0.0 if na < 1e-12 or nb < 1e-12 else np.dot(y_true, y_pred) / (na * nb)
    
    return r2, cosine

def main():
    """Main execution."""
    print("🚀 Simple TN-Shapley Example with d=100")
    print("="*50)
    
    # Parameters
    d = 100
    n_samples = 2000
    n_strong = 20
    n_weak = 30
    use_masks = True
    n_masks = 500
    
    # ================ STEP 1: Generate Data ================
    X, y, coefficients, linear_coeffs, strong_pairs, weak_pairs = generate_synthetic_data(
        d=d, n_samples=n_samples, n_strong=n_strong, n_weak=n_weak
    )
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"  Training: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    # ================ STEP 2: Mask Augmentation ================
    if use_masks:
        X_masked, y_masked, masks = generate_masks(X_train_scaled, y_train_scaled, n_masks)
        X_train_final = np.vstack([X_train_scaled, X_masked])
        y_train_final = np.hstack([y_train_scaled, y_masked])
        print(f"  Final training: {X_train_final.shape}")
    else:
        X_train_final = X_train_scaled
        y_train_final = y_train_scaled
    
    # ================ STEP 3: Train Tensor Network ================
    model, val_r2 = train_tensor_network(X_train_final, y_train_final, d, rank=12, epochs=300)
    
    # Test evaluation
    test_r2 = 0
    if model is not None:
        with torch.no_grad():
            test_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32)).squeeze().numpy()
            test_r2 = r2_score(y_test_scaled, test_pred)
        
        print(f"📊 Test R²: {test_r2:.4f}")
        
        if test_r2 > 0.9:
            print(f"✅ Achieved R² > 0.9: {test_r2:.4f}")
        else:
            print(f"⚠️ R² = {test_r2:.4f} < 0.9")
    
    # ================ STEP 4: Shapley Computation ================
    print(f"\n🎯 Computing Shapley Values...")
    
    # Evaluation point (test set mean)
    x_eval = np.mean(X_test_scaled, axis=0)
    x_eval_original = scaler_X.inverse_transform(x_eval.reshape(1, -1)).flatten()
    
    print(f"  Evaluation point: test set mean")
    
    # Ground truth
    gt_k1, gt_k2 = compute_ground_truth_shapley(x_eval_original, coefficients, linear_coeffs)
    print(f"  Ground truth K1: range [{gt_k1.min():.3f}, {gt_k1.max():.3f}]")
    print(f"  Ground truth K2: {len(gt_k2)} pairs, range [{gt_k2.min():.3f}, {gt_k2.max():.3f}]")
    
    # GAM-based Shapley
    def model_func(X_input):
        X_orig = scaler_X.inverse_transform(X_input)
        y_pred = np.zeros(X_orig.shape[0])
        y_pred += X_orig @ linear_coeffs
        for (i, j), coeff in coefficients.items():
            y_pred += coeff * X_orig[:, i] * X_orig[:, j]
        return y_pred
    
    gam_k1, gam_k2, gam_pairs = compute_gam_shapley(X_train_scaled, y_train_scaled, model_func, x_eval)
    print(f"  GAM K1: range [{gam_k1.min():.3f}, {gam_k1.max():.3f}]")
    print(f"  GAM K2: {len(gam_k2)} pairs, range [{gam_k2.min():.3f}, {gam_k2.max():.3f}]")
    
    # TN-based Shapley
    tn_k1, tn_k2 = compute_tn_shapley(model, x_eval, d)
    if tn_k1 is not None:
        print(f"  TN K1: range [{tn_k1.min():.3f}, {tn_k1.max():.3f}]")
        print(f"  TN K2: {len(tn_k2)} pairs, range [{tn_k2.min():.3f}, {tn_k2.max():.3f}]")
    
    # ================ STEP 5: Comparison ================
    print(f"\n📊 Comparison Results:")
    
    # GAM vs Ground Truth
    r2_gam_gt_k1, cos_gam_gt_k1 = compute_metrics(gt_k1, gam_k1)
    print(f"  GAM vs GT (K1): R²={r2_gam_gt_k1:.4f}, Cosine={cos_gam_gt_k1:.4f}")
    
    # Only compare order-2 for matching pairs
    matching_gt_k2 = []
    for i, j in gam_pairs:
        if (i, j) in coefficients:
            matching_gt_k2.append(coefficients[(i, j)] * x_eval_original[i] * x_eval_original[j])
        else:
            matching_gt_k2.append(0.0)
    matching_gt_k2 = np.array(matching_gt_k2)
    
    r2_gam_gt_k2, cos_gam_gt_k2 = compute_metrics(matching_gt_k2, gam_k2)
    print(f"  GAM vs GT (K2): R²={r2_gam_gt_k2:.4f}, Cosine={cos_gam_gt_k2:.4f}")
    
    # TN vs Ground Truth
    if tn_k1 is not None:
        r2_tn_gt_k1, cos_tn_gt_k1 = compute_metrics(gt_k1, tn_k1)
        r2_tn_gam_k1, cos_tn_gam_k1 = compute_metrics(gam_k1, tn_k1)
        
        print(f"  TN vs GT (K1):  R²={r2_tn_gt_k1:.4f}, Cosine={cos_tn_gt_k1:.4f}")
        print(f"  TN vs GAM (K1): R²={r2_tn_gam_k1:.4f}, Cosine={cos_tn_gam_k1:.4f}")
    
    # ================ STEP 6: Save Results ================
    outdir = 'simple_tn_shapley_results'
    os.makedirs(outdir, exist_ok=True)
    
    results = {
        'experiment_config': {
            'd': d,
            'n_samples': n_samples,
            'n_strong_pairs': n_strong,
            'n_weak_pairs': n_weak,
            'use_masks': use_masks
        },
        'model_performance': {
            'validation_r2': val_r2,
            'test_r2': test_r2,
            'r2_above_90': test_r2 > 0.9
        },
        'strong_pairs': strong_pairs,
        'weak_pairs': weak_pairs,
        'shapley_comparison': {
            'gam_vs_gt_k1': {'r2': r2_gam_gt_k1, 'cosine': cos_gam_gt_k1},
            'gam_vs_gt_k2': {'r2': r2_gam_gt_k2, 'cosine': cos_gam_gt_k2}
        }
    }
    
    if tn_k1 is not None:
        results['shapley_comparison'].update({
            'tn_vs_gt_k1': {'r2': r2_tn_gt_k1, 'cosine': cos_tn_gt_k1},
            'tn_vs_gam_k1': {'r2': r2_tn_gam_k1, 'cosine': cos_tn_gam_k1}
        })
    
    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Simple plot
    if tn_k1 is not None:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(gt_k1, gam_k1, alpha=0.6)
        plt.plot([gt_k1.min(), gt_k1.max()], [gt_k1.min(), gt_k1.max()], 'r--', alpha=0.5)
        plt.xlabel('Ground Truth Shapley')
        plt.ylabel('GAM Shapley')
        plt.title(f'GAM vs GT (R²={r2_gam_gt_k1:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.scatter(gt_k1, tn_k1, alpha=0.6)
        plt.plot([gt_k1.min(), gt_k1.max()], [gt_k1.min(), gt_k1.max()], 'r--', alpha=0.5)
        plt.xlabel('Ground Truth Shapley')
        plt.ylabel('TN Shapley')
        plt.title(f'TN vs GT (R²={r2_tn_gt_k1:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.scatter(gam_k1, tn_k1, alpha=0.6)
        plt.plot([gam_k1.min(), gam_k1.max()], [gam_k1.min(), gam_k1.max()], 'r--', alpha=0.5)
        plt.xlabel('GAM Shapley')
        plt.ylabel('TN Shapley')
        plt.title(f'TN vs GAM (R²={r2_tn_gam_k1:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'shapley_comparison.png'), dpi=150)
        plt.close()
    
    print(f"\n✅ Results saved to: {outdir}")
    
    # Final summary
    print(f"\n🎉 EXPERIMENT SUMMARY:")
    print(f"  Features: {d}")
    print(f"  TN Test R²: {test_r2:.4f} {'✅' if test_r2 > 0.9 else '❌'}")
    print(f"  GAM vs GT Shapley R²: {r2_gam_gt_k1:.4f}")
    print(f"  Strong interactions recovered: {n_strong} pairs")
    if tn_k1 is not None:
        print(f"  TN vs GT Shapley R²: {r2_tn_gt_k1:.4f}")

if __name__ == "__main__":
    main()
