"""
Configurable Neural Network Trainer
All hyperparameters and the dataset file are passed via command line.

Example:
    python train.py --data data.csv --target-col label \
        --hidden-layers 64 32 --activation relu \
        --epochs 50 --batch-size 32 --lr 0.001 \
        --optimizer adam --loss cross_entropy \
        --dropout 0.2 --val-split 0.2 --task classification \
        --output-model model.pt
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


def parse_args():
    p = argparse.ArgumentParser(description="Train a configurable neural network")

    # Data
    p.add_argument("--data", type=str, required=True,
                    help="Path to input data file (.csv)")
    p.add_argument("--target-col", type=str, required=True,
                    help="Name of the target/label column")
    p.add_argument("--val-split", type=float, default=0.2,
                    help="Fraction of data used for validation (default: 0.2)")

    # Architecture
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 32],
                    help="Sizes of hidden layers, e.g. --hidden-layers 128 64 32")
    p.add_argument("--activation", type=str, default="relu",
                    choices=["relu", "tanh", "sigmoid", "leaky_relu", "gelu"],
                    help="Activation function (default: relu)")
    p.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout probability (default: 0.0)")
    p.add_argument("--batch-norm", action="store_true",
                    help="Use batch normalization between layers")

    # Task
    p.add_argument("--task", type=str, default="classification",
                    choices=["classification", "regression"],
                    help="Task type (default: classification)")
    p.add_argument("--num-classes", type=int, default=None,
                    help="Number of classes for classification (auto-detected if omitted)")
    p.add_argument("--drop-cols", type=str, nargs="*", default=[],
                    help="Column names to drop from features (e.g. --drop-cols user_id)")

    # Training
    p.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs (default: 20)")
    p.add_argument("--batch-size", type=int, default=32,
                    help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (default: 1e-3)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                    help="Weight decay / L2 regularization (default: 0.0)")
    p.add_argument("--optimizer", type=str, default="adam",
                    choices=["adam", "sgd", "rmsprop", "adamw"],
                    help="Optimizer (default: adam)")
    p.add_argument("--momentum", type=float, default=0.9,
                    help="Momentum for SGD/RMSprop (default: 0.9)")
    p.add_argument("--loss", type=str, default=None,
                    choices=["cross_entropy", "mse", "mae", "bce"],
                    help="Loss function (auto-selected based on task if omitted)")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"], help="Device to train on")
    p.add_argument("--output-model", type=str, default="model.pt",
                    help="Path to save the trained model (default: model.pt)")
    p.add_argument("--log-interval", type=int, default=1,
                    help="Print training progress every N epochs (default: 1)")

    return p.parse_args()


def get_activation(name):
    return {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
    }[name]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation, dropout, batch_norm):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_optimizer(name, params, lr, weight_decay, momentum):
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")


def get_loss(name, task, num_classes):
    if name is None:
        name = "cross_entropy" if task == "classification" else "mse"
    return {
        "cross_entropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "bce": nn.BCEWithLogitsLoss(),
    }[name]


def load_data(path, target_col, drop_cols):
    if not path.lower().endswith(".csv"):
        print("Error: only .csv files are supported.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if target_col not in df.columns:
        print(f"Error: target column '{target_col}' not found in data.", file=sys.stderr)
        sys.exit(1)

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
        else:
            print(f"Warning: drop column '{col}' not found, skipping.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode non-numeric features
    X = pd.get_dummies(X)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True))
    X = X.fillna(0)

    return X.values.astype(np.float32), y


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    X, y_raw = load_data(args.data, args.target_col, args.drop_cols)

    if args.task == "classification":
        if not pd.api.types.is_numeric_dtype(y_raw):
            classes, y = np.unique(y_raw.astype(str), return_inverse=True)
            print("Class label mapping:")
            for idx, cls in enumerate(classes):
                print(f"  {idx} -> {cls}")
        else:
            y = y_raw.values.astype(np.int64)
            classes = np.unique(y)
        num_classes = args.num_classes or len(classes)
        y = y.astype(np.int64)
        output_dim = num_classes
    else:
        y = y_raw.values.astype(np.float32).reshape(-1, 1)
        output_dim = 1

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long if args.task == "classification" else torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_size > 0 else None

    model = MLP(
        input_dim=X.shape[1],
        hidden_layers=args.hidden_layers,
        output_dim=output_dim,
        activation=args.activation,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
    ).to(device)

    print(model)

    criterion = get_loss(args.loss, args.task, output_dim)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            if args.task == "regression" and isinstance(criterion, (nn.MSELoss, nn.L1Loss)):
                loss = criterion(preds, yb)
            else:
                loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / train_size

        val_msg = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
                    if args.task == "classification":
                        correct += (preds.argmax(dim=1) == yb).sum().item()
            val_loss /= val_size
            val_msg = f" | val_loss: {val_loss:.4f}"
            if args.task == "classification":
                val_msg += f" | val_acc: {correct/val_size:.4f}"

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f}{val_msg}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "input_dim": X.shape[1],
        "output_dim": output_dim,
        "classes": classes.tolist() if args.task == "classification" else None,
    }, args.output_model)
    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main()