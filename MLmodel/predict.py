"""
Run inference using a model trained by train.py

Example:
    python predict.py --model model.pt --data new_data.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description="Run inference with a trained model")
    p.add_argument("--model", type=str, required=True, help="Path to trained model (.pt)")
    p.add_argument("--data", type=str, required=True, help="Path to input CSV with feature columns")
    p.add_argument("--output", type=str, default="predictions.csv", help="Path to save predictions")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    classes = checkpoint.get("classes")

    model = MLP(
        input_dim=input_dim,
        hidden_layers=train_args["hidden_layers"],
        output_dim=output_dim,
        activation=train_args["activation"],
        dropout=train_args["dropout"],
        batch_norm=train_args["batch_norm"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if not args.data.lower().endswith(".csv"):
        print("Error: only .csv files are supported.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data)

    target_col = train_args["target_col"]
    drop_cols = train_args.get("drop_cols", [])

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    X = pd.get_dummies(df)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True))
    X = X.fillna(0)

    # Align columns with training input_dim (pad/truncate if mismatch)
    if X.shape[1] != input_dim:
        print(f"Warning: feature count mismatch (got {X.shape[1]}, expected {input_dim}). "
              "Make sure the input CSV has the same columns/categories as training data.", file=sys.stderr)
        if X.shape[1] < input_dim:
            for i in range(input_dim - X.shape[1]):
                X[f"_pad_{i}"] = 0
        else:
            X = X.iloc[:, :input_dim]

    X_tensor = torch.tensor(X.values.astype(np.float32)).to(device)

    with torch.no_grad():
        preds = model(X_tensor)

    task = train_args["task"]
    if task == "classification":
        pred_idx = preds.argmax(dim=1).cpu().numpy()
        if classes:
            pred_labels = [classes[i] for i in pred_idx]
            df["prediction"] = pred_labels
        else:
            df["prediction"] = pred_idx
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        df["confidence"] = probs.max(axis=1)
    else:
        df["prediction"] = preds.cpu().numpy().flatten()

    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(df.head())


if __name__ == "__main__":
    main()