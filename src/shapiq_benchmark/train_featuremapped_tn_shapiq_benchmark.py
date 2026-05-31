#!/usr/bin/env python3
"""
Train a FeatureMappedTN (MLP + BinaryTensorTree) on a shapiq benchmark Game.

Benchmarks:
  - AdultCensusDataValuation (15 players, SV benchmark in docs)
  - SentimentAnalysisLocalXAI (14 players, k-SII benchmark in docs)

We:
  1) Load a Game via `load_games_from_configuration`.
  2) Sample O(n) coalitions in {0,1}^n (train + val).
  3) Query v(S).
  4) Train FeatureMappedTN (MLP feature map + tensor tree) with Adam + ReduceLROnPlateau.
  5) Optionally: after `freeze_epoch`, freeze the MLP feature map and continue
     training only the tensor network part.
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from shapiq.benchmark import load_games_from_configuration
from feature_mapped_tn import make_feature_mapped_tn


# -----------------------------
# Helper utilities
# -----------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_random_coalitions(
    n_players: int,
    n_coalitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample random coalitions as {0,1}^n.

    Returns:
        coalitions: np.ndarray of shape [n_coalitions, n_players], dtype=bool
    """
    coalitions = rng.integers(
        low=0,
        high=2,
        size=(n_coalitions, n_players),
        dtype=np.int8,
    )
    return coalitions.astype(bool)


def game_values_to_tensor(
    game,
    coalitions: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate game on coalitions and convert to torch.float32 tensors.

    Args:
        game: shapiq Game – callable on coalition matrices (bool or {0,1}).
        coalitions: np.ndarray [B, n_players], dtype=bool.
        device: torch device.

    Returns:
        x: torch.FloatTensor [B, n_players] with entries in {0,1}.
        y: torch.FloatTensor [B, 1].
    """
    v = game(coalitions)  # [B] or [B,]
    v = np.asarray(v, dtype=np.float32).reshape(-1, 1)

    x = coalitions.astype(np.float32)
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    y_t = torch.from_numpy(v).to(device=device, dtype=torch.float32)
    return x_t, y_t


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    mean_true = torch.mean(y_true).item()
    ss_tot = torch.sum((y_true - mean_true) ** 2).item()
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# -----------------------------
# Core training routine
# -----------------------------

def train_featuremapped_tn_on_game(
    game_identifier: str,
    config_id: int,
    n_player_id: int,
    seed: int,
    n_train_factor: int,
    n_val_factor: int,
    epochs: int,
    batch_size: int,
    lr: float,
    rank: int,
    fmap_out_dim: int,
    fmap_hidden: int,
    use_polynomial_features: bool,
    save_path: str,
    device: str = "auto",
    freeze_epoch: int = 0,
):
    """
    Train FeatureMappedTN on a single shapiq benchmark Game instance.

    If freeze_epoch > 0, then after that epoch the MLP feature map (model.fmap)
    is frozen (requires_grad=False) and only the tensor network part continues
    training.
    """

    # ---- device ----
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    set_seed(seed)
    rng = np.random.default_rng(seed)

    # ---- load one Game from shapiq.benchmark ----
    games = load_games_from_configuration(
        game_class=game_identifier,
        n_player_id=n_player_id,
        config_id=config_id,
        n_games=1,
    )
    games = list(games)
    if len(games) == 0:
        raise RuntimeError("No games loaded from configuration.")
    game = games[0]
    n_players = game.n_players

    print(f"[INFO] Loaded Game: {game}")
    print(f"[INFO] n_players = {n_players}")

    # ---- choose sample sizes: O(n) coalitions ----
    n_train = max(n_players * n_train_factor, n_players)
    n_val = max(n_players * n_val_factor, n_players)

    print(
        f"[INFO] Sampling {n_train} train coalitions and {n_val} val coalitions "
        f"(train_factor={n_train_factor}, val_factor={n_val_factor})."
    )

    # ---- structured + random sampling for train, random for val ----
    def sample_structured_plus_random_coalitions(
        n_players: int,
        n_coalitions: int,
        rng: np.random.Generator,
        include_all_zero: bool = True,
        include_singletons: bool = True,
    ) -> np.ndarray:
        """
        Sample coalitions with a structured core + random tail.

        Structured part:
        - all-ones coalition
        - optionally all-zero coalition
        - for each feature i:
                * 'others 1, i = 0' coalition
                * optionally 'only i = 1, others 0' coalition

        Remaining coalitions are filled with uniform random {0,1}^n.

        Returns:
            coalitions: [n_coalitions, n_players], dtype=bool
        """
        base = []

        # (1) all-ones coalition
        all_ones = np.ones(n_players, dtype=bool)
        base.append(all_ones)

        # (2) optionally all-zero coalition
        if include_all_zero:
            all_zero = np.zeros(n_players, dtype=bool)
            base.append(all_zero)

        # (3) "others 1, i = 0" for each feature
        for i in range(n_players):
            v = np.ones(n_players, dtype=bool)
            v[i] = False
            base.append(v)

        # (4) optionally singleton coalitions: only i = 1
        if include_singletons:
            for i in range(n_players):
                v = np.zeros(n_players, dtype=bool)
                v[i] = True
                base.append(v)

        base = np.array(base, dtype=bool)
        n_base = base.shape[0]

        if n_base >= n_coalitions:
            # Just truncate if budget is tiny
            return base[:n_coalitions]

        # Remaining coalitions are random
        n_rand = n_coalitions - n_base
        rand_part = sample_random_coalitions(n_players, n_rand, rng)

        coalitions = np.concatenate([base, rand_part], axis=0)
        return coalitions

    coal_train = sample_structured_plus_random_coalitions(
        n_players=n_players,
        n_coalitions=n_train,
        rng=rng,
        include_all_zero=True,
        include_singletons=True,
    )
    # validation: purely random
    coal_val = sample_random_coalitions(n_players, n_val, rng)

    # ---- evaluate game on them ----
    x_train, y_train = game_values_to_tensor(game, coal_train, dev)
    x_val, y_val = game_values_to_tensor(game, coal_val, dev)

    print(f"[INFO] x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"[INFO] x_val   shape: {x_val.shape},   y_val   shape: {y_val.shape}")

    # ---- build FeatureMappedTN model ----
    print("[INFO] Building FeatureMappedTN model...")
    model = make_feature_mapped_tn(
        d_in=n_players,
        fmap_out_dim=fmap_out_dim,
        ranks=rank,
        out_dim=1,
        fmap_hidden=fmap_hidden,
        fmap_act="relu",
        use_polynomial_features=use_polynomial_features,
        use_log_scale=False,
        selector_mode="none",
        seed=seed,
        device=dev,
        dtype=torch.float32,
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model built. Total parameters: {n_params}")

    # ---- training setup ----
    dataset_train = TensorDataset(x_train, y_train)
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    criterion = nn.MSELoss()

    def make_optimizer_and_scheduler():
        # only include parameters with requires_grad=True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=20,
            min_lr=1e-7,
            verbose=True,
        )
        return opt, sch

    optimizer, scheduler = make_optimizer_and_scheduler()

    # ---- training loop ----
    print("[INFO] Starting training...")
    if freeze_epoch > 0:
        print(f"[INFO] Will freeze MLP feature map after epoch {freeze_epoch}.")

    start_time = time.time()
    mlp_frozen = False

    for epoch in range(1, epochs + 1):
        # freeze MLP after freeze_epoch (once)
        if (not mlp_frozen) and (freeze_epoch > 0) and (epoch == freeze_epoch + 1):
            print(f"[INFO] Freezing MLP feature map at start of epoch {epoch}.")
            if hasattr(model, "fmap"):
                for p in model.fmap.parameters():
                    p.requires_grad = False
            else:
                print("[WARN] model has no attribute 'fmap'; cannot freeze MLP.")
            optimizer, scheduler = make_optimizer_and_scheduler()
            mlp_frozen = True

        model.train()
        running_loss = 0.0
        n_seen = 0

        for xb, yb in loader_train:
            optimizer.zero_grad()
            pred = model(xb)        # [B, 1]
            if pred.ndim == 1:
                pred = pred.unsqueeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        train_loss = running_loss / max(n_seen, 1)

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            yhat_val = model(x_val)
            if yhat_val.ndim == 1:
                yhat_val = yhat_val.unsqueeze(-1)
            val_loss = criterion(yhat_val, y_val).item()
            val_r2 = r2_score(y_val, yhat_val)

        # step scheduler based on validation loss
        scheduler.step(val_loss)

        if epoch % max(1, epochs // 20) == 0 or epoch == 1 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch:4d}/{epochs}] "
                f"train_loss={train_loss:.4e}  "
                f"val_loss={val_loss:.4e}  "
                f"val_R2={val_r2:.4f}  "
                f"lr={current_lr:.2e}"
            )

    total_time = time.time() - start_time
    print(f"[INFO] Training finished in {total_time:.2f} seconds.")

    # ---- final evaluation ----
    model.eval()
    with torch.no_grad():
        yhat_train = model(x_train)
        if yhat_train.ndim == 1:
            yhat_train = yhat_train.unsqueeze(-1)
        yhat_val = model(x_val)
        if yhat_val.ndim == 1:
            yhat_val = yhat_val.unsqueeze(-1)

    train_r2 = r2_score(y_train, yhat_train)
    val_r2 = r2_score(y_val, yhat_val)
    print(f"[RESULT] Train R²: {train_r2:.4f}")
    print(f"[RESULT] Val   R²: {val_r2:.4f}")

    # ---- save model & metadata ----
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "n_players": n_players,
            "game_identifier": game_identifier,
            "config_id": config_id,
            "n_player_id": n_player_id,
            "seed": seed,
            "rank": rank,
            "fmap_out_dim": fmap_out_dim,
            "fmap_hidden": fmap_hidden,
            "use_polynomial_features": use_polynomial_features,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "freeze_epoch": freeze_epoch,
        }
        torch.save(checkpoint, save_path)
        print(f"[INFO] Saved model checkpoint to: {save_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FeatureMappedTN on a shapiq benchmark Game "
                    "(AdultCensusDataValuation or SentimentAnalysisLocalXAI)."
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["adult_sv", "sentiment_si"],
        required=True,
        help=(
            "Which shapiq benchmark to use:\n"
            "  adult_sv     -> AdultCensusDataValuation (SV benchmark example)\n"
            "  sentiment_si -> SentimentAnalysisLocalXAI (k-SII benchmark example)"
        ),
    )

    # training regime: O(n) coalitions
    parser.add_argument("--n_train_factor", type=int, default=20,
                        help="Number of train coalitions = n_players * n_train_factor.")
    parser.add_argument("--n_val_factor", type=int, default=20,
                        help="Number of val coalitions = n_players * n_val_factor.")

    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Initial learning rate.")

    # TN + feature map hyperparams
    parser.add_argument("--rank", type=int, default=4,
                        help="Tensor tree rank (same on all edges).")
    parser.add_argument("--fmap_out_dim", type=int, default=1,
                        help="Feature map output channels per feature (ignored if --use_polynomial_features).")
    parser.add_argument("--fmap_hidden", type=int, default=8,
                        help="Hidden size of the elementwise MLP feature map.")
    parser.add_argument("--use_polynomial_features", action="store_true",
                        help="Use fixed [x^2, x] features instead of MLP.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cpu', or 'cuda'.")

    parser.add_argument("--save_path", type=str, default="tn_shapiq_model.pt",
                        help="Path to save trained model checkpoint.")

    parser.add_argument("--freeze_epoch", type=int, default=0,
                        help="Epoch after which to freeze MLP feature map. "
                             "0 means never freeze.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.benchmark == "adult_sv":
        game_identifier = "AdultCensusDataValuation"
        config_id = 1
        n_player_id = 0
    elif args.benchmark == "sentiment_si":
        game_identifier = "SentimentAnalysisLocalXAI"
        config_id = 1
        n_player_id = 0
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    train_featuremapped_tn_on_game(
        game_identifier=game_identifier,
        config_id=config_id,
        n_player_id=n_player_id,
        seed=args.seed,
        n_train_factor=args.n_train_factor,
        n_val_factor=args.n_val_factor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rank=args.rank,
        fmap_out_dim=args.fmap_out_dim,
        fmap_hidden=args.fmap_hidden,
        use_polynomial_features=args.use_polynomial_features,
        save_path=args.save_path,
        device=args.device,
        freeze_epoch=args.freeze_epoch,
    )
