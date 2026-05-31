import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.tntree_model import BinaryTensorTree
from src.feature_mapped_tn import FeatureMappedTN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_tn_on_masked(
    masked_X: np.ndarray,
    y_teacher: np.ndarray,
    ranks: int,
    seed: int,
    max_epochs=100,
    tol=1e-6,
    lr=2e-2,
    amp=True,
    batch_size=100,
    fmap_hidden: int = 32,
    fmap_out_dim: int = 4,
    print_every: int = 50,
    patience: int = 200,
    target_r2: float = None,
    compile_model: bool = False,
    verbose_debug: bool = True,
    test_split: float = 0.0,
):
    """
    Train FeatureMappedTN (BinaryTensorTree + feature map) as TN surrogate
    on the masked Chebyshev dataset.
    
    Args:
        masked_X: Input features [N, d]
        y_teacher: Target outputs [N]
        ranks: Tensor network ranks
        seed: Random seed
        max_epochs: Maximum training epochs
        tol: Tolerance for early stopping
        lr: Learning rate
        amp: Whether to use automatic mixed precision
        batch_size: Batch size
        fmap_hidden: Hidden layer size in feature map
        print_every: Print frequency
        patience: Patience for early stopping
        target_r2: Target R² for early stopping
        compile_model: Whether to compile model
        verbose_debug: Whether to print detailed debug info
    """
    if BinaryTensorTree is None:
        raise RuntimeError("tntree_model.BinaryTensorTree not found.")

    set_all_seeds(seed)
    dev = DEVICE

    # Train/test split if requested
    n_total = len(masked_X)
    if test_split > 0:
        n_test = int(n_total * test_split)
        n_train = n_total - n_test
        indices = np.random.RandomState(seed).permutation(n_total)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train, y_train = masked_X[train_idx], y_teacher[train_idx]
        X_test, y_test = masked_X[test_idx], y_teacher[test_idx]
        
        print(f"[Split] Train: {n_train} samples, Test: {n_test} samples ({test_split*100:.1f}%)")
    else:
        X_train, y_train = masked_X, y_teacher
        X_test, y_test = None, None

    X_t = torch.tensor(X_train, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    n, d = X_t.shape
    
    # Normalize targets for better numerical stability (use training stats)
    y_mean = y_t.mean()
    y_std = y_t.std().clamp_min(1e-8)
    y_normalized = (y_t - y_mean) / y_std
    
    # Convert test set if available
    if X_test is not None:
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=dev)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=dev)
        y_test_normalized = (y_test_t - y_mean) / y_std
    else:
        X_test_t = None
        y_test_normalized = None
    
    if verbose_debug:
        print("[Debug] Raw y_teacher: mean={:.4f}, std={:.4f}".format(
            float(y_mean.item()), float(y_std.item())))
        print("[Debug] Input X: mean={:.4f}, std={:.4f}".format(
            float(X_t.mean().item()), float(X_t.std().item())))
        print("[Debug] Normalized y: mean={:.4f}, std={:.4f}".format(
            float(y_normalized.mean().item()), float(y_normalized.std().item())))

    from src.feature_mapped_tn import make_feature_mapped_tn

    print("[Debug] About to create model...", flush=True)
    # build TN surrogate with your canonical builder
    # Use LEARNED MLP feature map (polynomial features don't work with many features)
    model = make_feature_mapped_tn(
        d_in=d,
        fmap_out_dim=fmap_out_dim,  # Number of feature map output channels
        ranks=ranks,
        out_dim=1,
        fmap_hidden=fmap_hidden,     # Hidden layer size for MLP
        fmap_act="tanh",             # Activation function
        use_polynomial_features=False,  # Use learned MLP (NOT polynomial features)
        use_log_scale=False,          # Disable log-scale for now (has output shape bug)
        selector_mode="none",
        seed=seed,
        device=dev,
        dtype=torch.float32,
    ).to(dev)
    print("[Debug] Model created!", flush=True)
    
    # ---- INITIALIZATION: Simple uniform initialization ---- #
    print("[Debug] Applying simple uniform initialization...", flush=True)
    
    # Calculate tree depth for reference
    tree_depth = int(np.ceil(np.log2(d)))
    print(f"[Debug] Tree depth ≈ {tree_depth} levels", flush=True)
    
    with torch.no_grad():
        # Simple uniform initialization scaled for tree depth
        # Use smaller values for deeper trees to prevent explosion/vanishing
        init_scale = 0.5 / np.sqrt(tree_depth)
        
        print(f"[Debug] Using init scale: {init_scale:.4f}", flush=True)
        
        for name, p in model.tn.named_parameters():
            if p.ndim >= 2:  # Core tensors
                p.uniform_(-init_scale, init_scale)
        
        # Check initial outputs (should be small but nonzero)
        n_test = min(256, n)
        X_test = X_t[:n_test]
        model.eval()
        y_init = model(X_test).squeeze(-1)
        
        print(f"[Debug] Initial output range: [{y_init.min().item():.6f}, {y_init.max().item():.6f}]", flush=True)
        print(f"[Debug] Initial output std: {y_init.std().item():.6f}", flush=True)
    
    print("[Debug] Model initialized!", flush=True)
    
    # Check feature map output and model predictions
    print("[Debug] Checking feature map output...", flush=True)
    with torch.no_grad():
        sample_X = X_t[:10]  # Take first 10 samples
        features = model.feature_map(sample_X)  # Should be [10, d, 2]
        print(f"[Debug] Sample input X shape: {sample_X.shape}, range: [{sample_X.min():.4f}, {sample_X.max():.4f}]", flush=True)
        print(f"[Debug] Feature map output shape: {features.shape}", flush=True)
        print(f"[Debug] Features range: [{features.min():.4f}, {features.max():.4f}]", flush=True)
        print(f"[Debug] Features mean: {features.mean():.4f}, std: {features.std():.4f}", flush=True)
        
        # Check full model output
        model_out = model(sample_X).squeeze(-1)
        print(f"[Debug] Model output shape: {model_out.shape}", flush=True)
        print(f"[Debug] Model output range: [{model_out.min():.6f}, {model_out.max():.6f}]", flush=True)
        print(f"[Debug] Model output mean: {model_out.mean():.6f}, std: {model_out.std():.6f}", flush=True)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore
        except Exception:
            pass

    if batch_size is None:
        if n <= 2048:
            batch_size = n
        else:
            batch_size = min(16384, max(512, n // 16))

    # Use AdamW with stable hyperparameters for high R²
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, eps=1e-8)
    
    # Linear warmup for first 10 epochs to avoid early collapse
    warmup_epochs = 10
    base_lr = lr * 0.1  # Start at 10% of target LR
    
    # ReduceLROnPlateau for adaptive learning rate - more aggressive for faster convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=20, verbose=verbose_debug, min_lr=lr * 0.001
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and dev.type == "cuda"))
    crit = nn.MSELoss()

    idx = torch.arange(n, device=dev)
    best_state = None
    best_mse = float("inf")
    noimp = 0
    t0 = time.perf_counter()

    var_y = torch.var(y_normalized, unbiased=True).clamp_min(1e-12)
    if X_test_t is not None:
        var_y_test = torch.var(y_test_normalized, unbiased=True).clamp_min(1e-12)

    def eval_full():
        """Evaluate on training and test datasets."""
        model.eval()
        with torch.no_grad():
            # Training metrics
            pred_raw = model(X_t).squeeze(-1)
            mse_train = crit(pred_raw, y_normalized).item()
            r2_train = 1.0 - ((pred_raw - y_normalized).pow(2).mean() / var_y).item()
            pred_mean = float(pred_raw.mean().item())
            pred_std = float(pred_raw.std().item())
            
            # Test metrics if available
            if X_test_t is not None:
                pred_test = model(X_test_t).squeeze(-1)
                mse_test = crit(pred_test, y_test_normalized).item()
                r2_test = 1.0 - ((pred_test - y_test_normalized).pow(2).mean() / var_y_test).item()
            else:
                mse_test, r2_test = None, None
                
        return mse_train, r2_train, pred_mean, pred_std, mse_test, r2_test

    print("[TN] Training started for R² > 0.9...", flush=True)
    print(f"[TN] Model has {sum(p.numel() for p in model.parameters())} parameters", flush=True)
    print(f"[TN] Starting epoch loop...", flush=True)
    print(f"[Debug] Batch size: {batch_size}, Train size: {n}", flush=True)
    print(f"[Debug] About to enter training loop...", flush=True)
    for ep in range(1, max_epochs + 1):
        print(f"[Debug] Epoch {ep} started", flush=True)
        # Linear warmup for first few epochs
        if ep <= warmup_epochs:
            warmup_factor = ep / warmup_epochs
            current_lr = base_lr + (lr - base_lr) * warmup_factor
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr
        print(f"[Debug] Warmup done, setting model to train mode", flush=True)
        
        model.train()
        print(f"[Debug] Model in train mode, batch_size={batch_size}, n={n}", flush=True)

        if batch_size >= n:
            # Full batch training
            print(f"[Debug] Full batch training - about to zero grad", flush=True)
            opt.zero_grad(set_to_none=True)
            print(f"[Debug] About to forward pass with X_t shape {X_t.shape}", flush=True)
            with torch.cuda.amp.autocast(enabled=(amp and dev.type == "cuda")):
                pred = model(X_t).squeeze(-1)
                loss = crit(pred, y_normalized)
            print(f"[Debug] Forward done, loss={loss.item():.4f}", flush=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Conservative clipping
            scaler.step(opt)
            scaler.update()
            mse_epoch = loss.item()
        else:
            # Mini-batch training
            perm = idx[torch.randperm(n, device=dev)]
            total_loss = 0.0
            seen = 0
            max_grad_norm = 0.0
            
            for s in range(0, n, batch_size):
                sel = perm[s : s + batch_size]
                xb = X_t.index_select(0, sel)
                yb = y_normalized.index_select(0, sel)
                
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(amp and dev.type == "cuda")):
                    pred = model(xb).squeeze(-1)
                    loss = crit(pred, yb)
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Conservative clipping
                max_grad_norm = max(max_grad_norm, grad_norm)
                scaler.step(opt)
                scaler.update()
                
                total_loss += loss.item() * xb.size(0)
                seen += xb.size(0)
            
            mse_epoch = total_loss / max(seen, 1)
            grad_norm = max_grad_norm

        improved = mse_epoch < best_mse - 1e-12
        if improved:
            best_mse = mse_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1

        # Step scheduler with loss value
        scheduler.step(mse_epoch)

        if (ep == 1) or (ep % print_every == 0) or improved:
            mse_train, r2_train, pred_mean, pred_std, mse_test, r2_test = eval_full()
            
            if X_test_t is not None:
                print(
                    f"[TN] ep {ep:4d} | "
                    f"loss={mse_epoch:.4e} | "
                    f"best={best_mse:.4e} | "
                    f"train_R²={r2_train:+.4f} | "
                    f"test_R²={r2_test:+.4f} | "
                    f"pred_μ={pred_mean:+.4f} σ={pred_std:.4f} | "
                    f"∇={grad_norm:.4e}"
                )
            else:
                print(
                    f"[TN] ep {ep:4d} | "
                    f"loss={mse_epoch:.4e} | "
                    f"best={best_mse:.4e} | "
                    f"full_mse={mse_train:.4e} | "
                    f"R²={r2_train:+.4f} | "
                    f"pred_μ={pred_mean:+.4f} σ={pred_std:.4f} | "
                    f"∇={grad_norm:.4e}"
                )
            
            if target_r2 is not None and r2_train >= target_r2:
                print(f"[TN] ✓ Target R² {target_r2} reached at epoch {ep}")
                break

        if mse_epoch < tol:
            print(f"[TN] ✓ Tolerance {tol:.1e} reached at epoch {ep}")
            break
        if noimp > patience:
            print(f"[TN] ⓘ Early stop: no improvement for {noimp} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.perf_counter() - t0
    final_mse_train, final_r2_train, _, _, final_mse_test, final_r2_test = eval_full()
    print(f"\n[TN] Training complete in {elapsed:.2f}s")
    if X_test_t is not None:
        print(f"[TN] Final Train: mse={final_mse_train:.4e}, R²={final_r2_train:.4f}")
        print(f"[TN] Final Test:  mse={final_mse_test:.4e}, R²={final_r2_test:.4f}")
    else:
        print(f"[TN] Final: mse={final_mse_train:.4e}, R²={final_r2_train:.4f}")
    
    info = dict(
        final_mse_train=final_mse_train,
        final_r2_train=final_r2_train,
        final_mse_test=final_mse_test if X_test_t is not None else None,
        final_r2_test=final_r2_test if X_test_t is not None else None,
        best_mse=best_mse,
        epochs=ep,
        elapsed=elapsed,
        y_mean=float(y_mean.item()),
        y_std=float(y_std.item()),
    )
    return model, info
