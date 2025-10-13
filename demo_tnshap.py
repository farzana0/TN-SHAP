#!/usr/bin/env python3
"""
TNShap Demo: Quick Start Example

This script demonstrates the core TNShap functionality with a simple example
that shows how to:
1. Train a tensor network surrogate model
2. Compute Shapley values using TNShap
3. Compare with exact Shapley values (for synthetic data)

Run this script to get a quick overview of TNShap capabilities.
"""

import torch
import numpy as np
import time
import sys
import os

# Add the repository root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Skipping visualizations.")

try:
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. Skipping diabetes example.")

# Import TNShap components
try:
    from src.tntree_model import BinaryTensorTree
    from src.feature_mapped_tn import make_feature_mapped_tn
    from src.utils.shapley_computation import compute_shapley_values_tnshap
    HAS_TNSHAP = True
except ImportError as e:
    HAS_TNSHAP = False
    print(f"Error importing TNShap components: {e}")
    print("Please ensure you have installed the package: pip install -e .")


def synthetic_example():
    """Demonstrate TNShap on a synthetic multilinear function."""
    print("🧪 Synthetic Function Example")
    print("=" * 50)
    
    # Generate synthetic multilinear function: f(x) = x1*x2 + x3*x4 + x1*x3*x5
    def synthetic_function(X):
        """Synthetic multilinear function with known Shapley values."""
        return X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + X[:, 0] * X[:, 2] * X[:, 4]
    
    # Generate training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    y = synthetic_function(X)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"Function: f(x) = x1*x2 + x3*x4 + x1*x3*x5")
    
    # Create and train tensor network
    print("\n🔧 Training Tensor Network...")
    model = make_feature_mapped_tn(
        d_in=n_features,
        fmap_out_dim=4,
        ranks=6,
        seed=42
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}, Loss: {loss.item():.6f}")
    
    # Evaluate training fit
    model.eval()
    with torch.no_grad():
        train_pred = model(X_tensor)
        train_r2 = 1 - torch.sum((y_tensor - train_pred) ** 2) / torch.sum((y_tensor - torch.mean(y_tensor)) ** 2)
        print(f"  Training R²: {train_r2.item():.4f}")
    
    # Compute Shapley values for a test point
    print("\n📊 Computing Shapley Values...")
    test_point = torch.FloatTensor([[0.5, -0.3, 0.8, -0.2, 0.1]])
    
    # TNShap computation
    start_time = time.time()
    shapley_tnshap = compute_shapley_values_tnshap(model, test_point, order=1)
    tnshap_time = time.time() - start_time
    
    print(f"  TNShap computation time: {tnshap_time:.4f}s")
    print(f"  Shapley values: {shapley_tnshap.flatten()}")
    
    # For synthetic function, we can compute exact Shapley values
    # For f(x) = x1*x2 + x3*x4 + x1*x3*x5, the exact Shapley values are:
    # φ1 = x2/2 + x3*x5/3, φ2 = x1/2, φ3 = x4/2 + x1*x5/3, φ4 = x3/2, φ5 = x1*x3/3
    x = test_point[0].numpy()
    exact_shapley = np.array([
        x[1]/2 + x[2]*x[4]/3,  # φ1
        x[0]/2,                 # φ2  
        x[3]/2 + x[0]*x[4]/3,  # φ3
        x[2]/2,                 # φ4
        x[0]*x[2]/3             # φ5
    ])
    
    print(f"  Exact Shapley values: {exact_shapley}")
    
    # Compute correlation
    correlation = np.corrcoef(shapley_tnshap.flatten(), exact_shapley)[0, 1]
    print(f"  Correlation with exact: {correlation:.4f}")
    
    return {
        'train_r2': train_r2.item(),
        'correlation': correlation,
        'computation_time': tnshap_time,
        'shapley_tnshap': shapley_tnshap.flatten(),
        'shapley_exact': exact_shapley
    }


def diabetes_example():
    """Demonstrate TNShap on the Diabetes dataset."""
    if not HAS_SKLEARN:
        print("\n⚠️  Diabetes example requires scikit-learn")
        return None
        
    print("\n🏥 Diabetes Dataset Example")
    print("=" * 50)
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {', '.join(diabetes.feature_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a "teacher" model (Random Forest)
    print("\n🎓 Training Teacher Model (Random Forest)...")
    teacher = RandomForestRegressor(n_estimators=100, random_state=42)
    teacher.fit(X_train_scaled, y_train)
    
    # Get teacher predictions
    y_train_pred = teacher.predict(X_train_scaled)
    teacher_r2 = teacher.score(X_test_scaled, y_test)
    print(f"  Teacher R²: {teacher_r2:.4f}")
    
    # Train tensor network surrogate
    print("\n🔧 Training Tensor Network Surrogate...")
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_pred).unsqueeze(1)
    
    model = make_feature_mapped_tn(
        d_in=X.shape[1],
        fmap_out_dim=4,
        ranks=8,
        seed=42
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:3d}, Loss: {loss.item():.6f}")
    
    # Evaluate surrogate fit
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor)
        surrogate_r2 = 1 - torch.sum((y_train_tensor - train_pred) ** 2) / torch.sum((y_train_tensor - torch.mean(y_train_tensor)) ** 2)
        print(f"  Surrogate R²: {surrogate_r2.item():.4f}")
    
    # Compute Shapley values for a test point
    print("\n📊 Computing Shapley Values...")
    test_idx = 0
    test_point = torch.FloatTensor(X_test_scaled[test_idx:test_idx+1])
    
    # TNShap computation
    start_time = time.time()
    shapley_tnshap = compute_shapley_values_tnshap(model, test_point, order=1)
    tnshap_time = time.time() - start_time
    
    print(f"  TNShap computation time: {tnshap_time:.4f}s")
    print(f"  Shapley values: {shapley_tnshap.flatten()}")
    
    # Show feature importance
    feature_importance = np.abs(shapley_tnshap.flatten())
    top_features = np.argsort(feature_importance)[-3:][::-1]
    
    print(f"  Top 3 most important features:")
    for i, feat_idx in enumerate(top_features):
        print(f"    {i+1}. {diabetes.feature_names[feat_idx]}: {shapley_tnshap[0, feat_idx]:.4f}")
    
    return {
        'teacher_r2': teacher_r2,
        'surrogate_r2': surrogate_r2.item(),
        'computation_time': tnshap_time,
        'shapley_values': shapley_tnshap.flatten(),
        'feature_names': diabetes.feature_names
    }


def plot_results(synthetic_results, diabetes_results):
    """Create visualization of the results."""
    if not HAS_MATPLOTLIB:
        print("⚠️  Visualization requires matplotlib")
        return None
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Synthetic: TNShap vs Exact
    axes[0, 0].scatter(synthetic_results['shapley_exact'], synthetic_results['shapley_tnshap'], alpha=0.7)
    axes[0, 0].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('Exact Shapley Values')
    axes[0, 0].set_ylabel('TNShap Values')
    axes[0, 0].set_title(f'Synthetic Function\nCorrelation: {synthetic_results["correlation"]:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Synthetic: Feature importance
    axes[0, 1].bar(range(5), np.abs(synthetic_results['shapley_tnshap']))
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('|Shapley Value|')
    axes[0, 1].set_title('Synthetic: Feature Importance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Diabetes: Feature importance
    feature_importance = np.abs(diabetes_results['shapley_values'])
    top_5_idx = np.argsort(feature_importance)[-5:]
    top_5_names = [diabetes_results['feature_names'][i] for i in top_5_idx]
    top_5_values = diabetes_results['shapley_values'][top_5_idx]
    
    axes[1, 0].barh(range(5), top_5_values)
    axes[1, 0].set_yticks(range(5))
    axes[1, 0].set_yticklabels(top_5_names)
    axes[1, 0].set_xlabel('Shapley Value')
    axes[1, 0].set_title('Diabetes: Top 5 Features')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance comparison
    metrics = ['Training R²', 'Computation Time (s)']
    synthetic_vals = [synthetic_results['train_r2'], synthetic_results['computation_time']]
    diabetes_vals = [diabetes_results['surrogate_r2'], diabetes_results['computation_time']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, synthetic_vals, width, label='Synthetic', alpha=0.8)
    axes[1, 1].bar(x + width/2, diabetes_vals, width, label='Diabetes', alpha=0.8)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tnshap_demo_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Results saved to 'tnshap_demo_results.png'")
    
    return fig


def main():
    """Main demonstration function."""
    print("🚀 TNShap Demonstration")
    print("=" * 60)
    print("This demo shows how to use TNShap for Shapley value computation")
    print("using tensor networks as surrogate models.\n")
    
    if not HAS_TNSHAP:
        print("❌ Cannot run demo: TNShap components not available.")
        print("Please install the package: pip install -e .")
        return
    
    try:
        # Run synthetic example
        synthetic_results = synthetic_example()
        
        # Run diabetes example if sklearn is available
        diabetes_results = None
        if HAS_SKLEARN:
            diabetes_results = diabetes_example()
        else:
            print("\n⚠️  Skipping diabetes example (scikit-learn not available)")
        
        # Create visualization if matplotlib is available
        if HAS_MATPLOTLIB and diabetes_results is not None:
            print("\n📊 Creating Visualization...")
            fig = plot_results(synthetic_results, diabetes_results)
        else:
            print("\n⚠️  Skipping visualization (matplotlib not available)")
        
        # Summary
        print("\n🎉 Demo Summary")
        print("=" * 50)
        print(f"Synthetic Function:")
        print(f"  • Training R²: {synthetic_results['train_r2']:.4f}")
        print(f"  • Shapley Correlation: {synthetic_results['correlation']:.4f}")
        print(f"  • Computation Time: {synthetic_results['computation_time']:.4f}s")
        
        if diabetes_results is not None:
            print(f"\nDiabetes Dataset:")
            print(f"  • Teacher R²: {diabetes_results['teacher_r2']:.4f}")
            print(f"  • Surrogate R²: {diabetes_results['surrogate_r2']:.4f}")
            print(f"  • Computation Time: {diabetes_results['computation_time']:.4f}s")
        
        print(f"\n✅ Demo completed successfully!")
        if HAS_MATPLOTLIB and diabetes_results is not None:
            print(f"📁 Check 'tnshap_demo_results.png' for visualizations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
