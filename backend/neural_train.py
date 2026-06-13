import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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
                # Use BatchNorm1d if there's more than 1 sample (safety check during training/eval)
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

def get_loss(name, task):
    if name is None or name == "none":
        name = "cross_entropy" if task == "classification" else "mse"
    return {
        "cross_entropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "bce": nn.BCEWithLogitsLoss(),
    }[name]

def preprocess_datasets(df_train, df_test, target_col, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
        
    # Drop specified columns
    for col in drop_cols:
        if col in df_train.columns:
            df_train = df_train.drop(columns=[col])
        if col in df_test.columns:
            df_test = df_test.drop(columns=[col])
            
    # Extract target
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in training dataset.")
        
    y_train_raw = df_train[target_col]
    X_train_raw = df_train.drop(columns=[target_col])
    
    # Make a copy of df_test and drop target if present there
    X_test_raw = df_test.copy()
    if target_col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[target_col])
        
    # Impute missing values
    for col in X_train_raw.columns:
        if pd.api.types.is_numeric_dtype(X_train_raw[col]):
            X_train_raw[col] = X_train_raw[col].fillna(X_train_raw[col].median())
        else:
            mode_val = X_train_raw[col].mode()
            fill_val = mode_val[0] if len(mode_val) else "missing"
            X_train_raw[col] = X_train_raw[col].fillna(fill_val)
            
    for col in X_test_raw.columns:
        if col in X_train_raw.columns:
            # fill with train values for consistency
            if pd.api.types.is_numeric_dtype(X_train_raw[col]):
                X_test_raw[col] = X_test_raw[col].fillna(X_train_raw[col].median())
            else:
                mode_val = X_train_raw[col].mode()
                fill_val = mode_val[0] if len(mode_val) else "missing"
                X_test_raw[col] = X_test_raw[col].fillna(fill_val)
        else:
            if pd.api.types.is_numeric_dtype(X_test_raw[col]):
                X_test_raw[col] = X_test_raw[col].fillna(X_test_raw[col].median())
            else:
                mode_val = X_test_raw[col].mode()
                fill_val = mode_val[0] if len(mode_val) else "missing"
                X_test_raw[col] = X_test_raw[col].fillna(fill_val)

    # Get dummies for categorical features
    X_train_encoded = pd.get_dummies(X_train_raw)
    X_test_encoded = pd.get_dummies(X_test_raw)
    
    # Align test features with training features
    train_cols = X_train_encoded.columns.tolist()
    X_test_aligned = X_test_encoded.reindex(columns=train_cols, fill_value=0)
    
    # Convert to numeric float32
    X_train_final = X_train_encoded.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    X_test_final = X_test_aligned.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    
    return X_train_final, y_train_raw, X_test_final, train_cols

def train_and_predict(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    hidden_layers=[64, 32],
    activation="relu",
    dropout=0.0,
    batch_norm=False,
    task="classification",
    epochs=20,
    batch_size=32,
    lr=0.001,
    optimizer_name="adam",
    loss_name=None,
    val_split=0.2,
    seed=42,
    device="cpu",
    drop_cols=None
):
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Preprocess
    X_train, y_raw, X_test, feature_cols = preprocess_datasets(df_train, df_test, target_col, drop_cols)
    
    # Target conversion
    classes = None
    if task == "classification":
        if not pd.api.types.is_numeric_dtype(y_raw):
            classes_arr, y_train_labels = np.unique(y_raw.astype(str), return_inverse=True)
            classes = classes_arr.tolist()
        else:
            y_train_labels = y_raw.values.astype(np.int64)
            classes = [str(x) for x in np.unique(y_train_labels)]
        
        num_classes = len(classes)
        y_train = y_train_labels.astype(np.int64)
        output_dim = num_classes
    else:
        y_train = y_raw.values.astype(np.float32).reshape(-1, 1)
        output_dim = 1
        
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long if task == "classification" else torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_ds = dataset
        val_ds = None
        val_loader = None
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
    ).to(device)
    
    criterion = get_loss(loss_name, task)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay=0.0, momentum=0.9)
    
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            
        train_loss = total_loss / len(train_ds)
        
        val_loss = None
        val_acc = None
        
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    total_val_loss += loss.item() * xb.size(0)
                    if task == "classification":
                        correct += (preds.argmax(dim=1) == yb).sum().item()
            val_loss = total_val_loss / val_size
            if task == "classification":
                val_acc = correct / val_size
                
        epoch_data = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_acc": float(val_acc) if val_acc is not None else None
        }
        history.append(epoch_data)
        
    # Run predictions on df_test
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        test_preds = model(X_test_tensor)
        
    predictions_list = []
    
    # We construct a prediction output aligned with each row in df_test
    for i in range(len(df_test)):
        row_dict = df_test.iloc[i].to_dict()
        # Clean any float nan/inf values so JSON serializes correctly
        clean_row_dict = {}
        for k, v in row_dict.items():
            if pd.isna(v):
                clean_row_dict[k] = None
            else:
                clean_row_dict[k] = v
                
        if task == "classification":
            logits = test_preds[i]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            pred_idx = logits.argmax().item()
            pred_label = classes[pred_idx] if classes else int(pred_idx)
            confidence = float(probs[pred_idx])
            
            clean_row_dict["prediction"] = pred_label
            clean_row_dict["confidence"] = confidence
        else:
            pred_val = test_preds[i].item()
            clean_row_dict["prediction"] = float(pred_val)
            clean_row_dict["confidence"] = None
            
        predictions_list.append(clean_row_dict)
        
    return {
        "history": history,
        "predictions": predictions_list,
        "classes": classes,
        "features_count": X_train.shape[1],
        "samples_count": len(df_train),
        "test_samples_count": len(df_test),
        "task": task
    }
