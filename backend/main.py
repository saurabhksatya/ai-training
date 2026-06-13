from fastapi import FastAPI
import pandas as pd
from fastapi import UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
from train import train_models
from fastapi.middleware.cors import CORSMiddleware

class TrainingRequest(BaseModel):
    target_column: str

    models: List[str]

    rf_n_estimators: int = 100
    rf_max_depth: int = 10

    knn_neighbors: int = 5

    gb_estimators: int = 100

app = FastAPI()

MAX_FILE_SIZE_MB = 10
MAX_ROWS = 50000
MAX_COLS = 150
MAX_EPOCHS = 150
MAX_TREES = 500
MAX_LAYERS = 4
MAX_NODES_PER_LAYER = 512


async def validate_file_size(file: UploadFile):
    try:
        await file.seek(0, 2)
        size = await file.tell()
        await file.seek(0)
        max_size = MAX_FILE_SIZE_MB * 1024 * 1024
        if size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB."
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        pass


def validate_df_dimensions(df: pd.DataFrame, name: str):
    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"{name} dataset exceeds the maximum limit of {MAX_ROWS} rows."
        )
    if len(df.columns) > MAX_COLS:
        raise HTTPException(
            status_code=400,
            detail=f"{name} dataset exceeds the maximum limit of {MAX_COLS} columns."
        )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=False, # Must be False if allow_origins=["*"]
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/health")
async def health():
    return {"status": "ok"}

from sklearn.datasets import (
    load_iris,
    load_breast_cancer
)

@app.get("/datasets/iris")
def iris_dataset():
    data = load_iris()

    df = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    df["target"] = data.target

    return {
        "rows": len(df),
        "columns": list(df.columns)
    }


@app.get("/datasets/breast-cancer")
def breast_cancer_dataset():
    data = load_breast_cancer()

    df = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    return {
        "rows": len(df),
        "columns": list(df.columns)
    }

@app.post("/train")
async def train(
    config: str = Form(...),
    file: UploadFile = File(...)
):
    import json

    settings = TrainingRequest(**json.loads(config))

    # Validate file size
    await validate_file_size(file)

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)

    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )

    # Validate dimensions
    validate_df_dimensions(df, "Uploaded")

    # Validate parameters
    if settings.rf_n_estimators > MAX_TREES or settings.rf_n_estimators < 1:
        raise HTTPException(status_code=400, detail=f"Random Forest trees must be between 1 and {MAX_TREES}.")
    if settings.gb_estimators > MAX_TREES or settings.gb_estimators < 1:
        raise HTTPException(status_code=400, detail=f"Gradient Boosting estimators must be between 1 and {MAX_TREES}.")
    if settings.rf_max_depth > 30 or settings.rf_max_depth < 1:
        raise HTTPException(status_code=400, detail="Random Forest max depth must be between 1 and 30.")
    if settings.knn_neighbors > 50 or settings.knn_neighbors < 1:
        raise HTTPException(status_code=400, detail="k-NN neighbors must be between 1 and 50.")

    result = train_models(
        df=df,
        target_variable=settings.target_column,
        selected_models=settings.models,
        rf_n_estimators=settings.rf_n_estimators,
        rf_max_depth=settings.rf_max_depth,
        knn_neighbors=settings.knn_neighbors,
        gb_estimators=settings.gb_estimators,
    )

    return result


class NeuralNetworkRequest(BaseModel):
    target_column: str
    hidden_layers: List[int] = [64, 32]
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False
    task: str = "classification"
    epochs: int = 20
    batch_size: int = 32
    lr: float = 0.001
    optimizer: str = "adam"
    loss: str = "none"
    val_split: float = 0.2
    drop_cols: List[str] = []


@app.post("/neural-train")
async def neural_train(
    config: str = Form(...),
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...)
):
    import json
    from neural_train import train_and_predict

    try:
        settings = NeuralNetworkRequest(**json.loads(config))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # Validate file sizes
    await validate_file_size(train_file)
    await validate_file_size(test_file)

    # Load train file
    if train_file.filename.endswith(".csv"):
        df_train = pd.read_csv(train_file.file)
    elif train_file.filename.endswith(".xlsx"):
        df_train = pd.read_excel(train_file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported training file type")

    # Load test file
    if test_file.filename.endswith(".csv"):
        df_test = pd.read_csv(test_file.file)
    elif test_file.filename.endswith(".xlsx"):
        df_test = pd.read_excel(test_file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported testing file type")

    # Validate dimensions
    validate_df_dimensions(df_train, "Training")
    validate_df_dimensions(df_test, "Testing")

    # Validate parameters
    if settings.epochs > MAX_EPOCHS or settings.epochs < 1:
        raise HTTPException(status_code=400, detail=f"Epochs must be between 1 and {MAX_EPOCHS}.")
    if len(settings.hidden_layers) > MAX_LAYERS:
        raise HTTPException(status_code=400, detail=f"Number of hidden layers cannot exceed {MAX_LAYERS}.")
    for i, nodes in enumerate(settings.hidden_layers):
        if nodes > MAX_NODES_PER_LAYER or nodes < 1:
            raise HTTPException(status_code=400, detail=f"Hidden layer {i+1} must have between 1 and {MAX_NODES_PER_LAYER} nodes.")
    if settings.dropout > 0.9 or settings.dropout < 0.0:
        raise HTTPException(status_code=400, detail="Dropout rate must be between 0.0 and 0.9.")
    if settings.val_split >= 0.5 or settings.val_split < 0.0:
        raise HTTPException(status_code=400, detail="Validation split must be between 0.0 and 0.5 (exclusive).")
    if settings.lr > 1.0 or settings.lr <= 0.0:
        raise HTTPException(status_code=400, detail="Learning rate must be positive and less than or equal to 1.0.")

    try:
        result = train_and_predict(
            df_train=df_train,
            df_test=df_test,
            target_col=settings.target_column,
            hidden_layers=settings.hidden_layers,
            activation=settings.activation,
            dropout=settings.dropout,
            batch_norm=settings.batch_norm,
            task=settings.task,
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            lr=settings.lr,
            optimizer_name=settings.optimizer,
            loss_name=settings.loss,
            val_split=settings.val_split,
            seed=42,
            device="auto",
            drop_cols=settings.drop_cols
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error running training/prediction: {str(e)}")
