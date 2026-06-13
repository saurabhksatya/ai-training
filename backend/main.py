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

app = FastAPI(title="NeuroForge")

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


# ----------------------------------------------------------------------
# Reinforcement Learning (RLTrain) Endpoints
# ----------------------------------------------------------------------
from fastapi.responses import StreamingResponse, FileResponse
from rl_train import start_training_task, get_training_task, cancel_training_task, MODELS_DIR, QNetwork
import asyncio
import queue

class RLTrainRequest(BaseModel):
    env: str = "CartPole-v1"
    episodes: int = 200
    max_steps: int = 500
    gamma: float = 0.99
    lr: float = 0.001
    batch_size: int = 64
    buffer_size: int = 10000
    min_buffer_size: int = 1000
    target_update_freq: int = 10
    optimizer: str = "adam"
    weight_decay: float = 0.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_layers: List[int] = [128, 128]
    activation: str = "relu"
    seed: int = 42
    reward_rules: List[dict] = []
    fail_reward_threshold: float = 0.0

@app.post("/rl/train")
async def rl_train_endpoint(req: RLTrainRequest):
    # Validate hidden layers limit
    if len(req.hidden_layers) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 hidden layers allowed.")
    for h in req.hidden_layers:
        if h < 1 or h > 512:
            raise HTTPException(status_code=400, detail="Hidden layer size must be between 1 and 512.")
            
    # Validate other parameters
    if req.episodes < 1 or req.episodes > 500:
        raise HTTPException(status_code=400, detail="Episodes must be between 1 and 500.")
    if req.max_steps < 1 or req.max_steps > 1000:
        raise HTTPException(status_code=400, detail="Max steps must be between 1 and 1000.")
    if req.buffer_size < 100 or req.buffer_size > 50000:
        raise HTTPException(status_code=400, detail="Buffer size must be between 100 and 50000.")
    if req.batch_size < 1 or req.batch_size > 256:
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 256.")
    if req.lr < 1e-6 or req.lr > 0.1:
        raise HTTPException(status_code=400, detail="Learning rate must be between 1e-6 and 0.1.")
    if req.gamma < 0.0 or req.gamma > 1.0:
        raise HTTPException(status_code=400, detail="Discount factor must be between 0.0 and 1.0.")
        
    task_id = start_training_task(req.model_dump())
    return {"task_id": task_id}


@app.get("/rl/progress/{task_id}")
async def rl_progress_endpoint(task_id: str):
    import json
    task = get_training_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    async def event_generator():
        # Yield historical data
        for h in task.history:
            yield f"data: {json.dumps({'type': 'progress', 'data': h})}\n\n"
            
        for log in task.logs:
            yield f"data: {json.dumps({'type': 'log', 'data': log})}\n\n"
            
        last_history_sent = len(task.history)
        while task.status == "running":
            # Consume new logs
            try:
                while True:
                    log_line = task.log_queue.get_nowait()
                    yield f"data: {json.dumps({'type': 'log', 'data': log_line})}\n\n"
            except queue.Empty:
                pass
                
            # Consume new progress data
            current_history_len = len(task.history)
            if current_history_len > last_history_sent:
                for i in range(last_history_sent, current_history_len):
                    yield f"data: {json.dumps({'type': 'progress', 'data': task.history[i]})}\n\n"
                last_history_sent = current_history_len
                
            await asyncio.sleep(0.2)
            
        # Drain remaining logs
        try:
            while True:
                log_line = task.log_queue.get_nowait()
                yield f"data: {json.dumps({'type': 'log', 'data': log_line})}\n\n"
        except queue.Empty:
            pass
            
        # Final status
        yield f"data: {json.dumps({'type': 'status', 'status': task.status, 'error': task.error_message})}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/rl/cancel/{task_id}")
async def rl_cancel_endpoint(task_id: str):
    success = cancel_training_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "cancelled"}


@app.get("/rl/download/{task_id}")
async def rl_download_endpoint(task_id: str):
    import os
    model_path = os.path.join(MODELS_DIR, f"{task_id}.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(model_path, filename="dqn_model.pt", media_type="application/octet-stream")


@app.get("/rl/test/stream/{task_id}")
async def rl_test_stream_endpoint(task_id: str, episodes: int = 3, max_steps: int = 500):
    import os
    import torch
    import gymnasium as gym
    import numpy as np
    from PIL import Image
    import io
    
    # Enforce testing limits on the server side
    episodes = min(max(int(episodes), 1), 5)
    max_steps = min(max(int(max_steps), 1), 1000)
    
    model_path = os.path.join(MODELS_DIR, f"{task_id}.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
        
    device = "cpu"
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        train_args = checkpoint["args"]
        state_dim = checkpoint["state_dim"]
        action_dim = checkpoint["action_dim"]
        
        # Instantiate network
        model = QNetwork(state_dim, train_args["hidden_layers"], action_dim, train_args["activation"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
    async def frame_generator():
        env = None
        try:
            env = gym.make(train_args["env"], render_mode="rgb_array")
            for ep in range(episodes):
                state, _ = env.reset()
                state = np.array(state, dtype=np.float32).flatten()
                
                for step in range(max_steps):
                    with torch.no_grad():
                        q_values = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                        action = int(q_values.argmax(dim=1).item())
                        
                    state, reward, terminated, truncated, _ = env.step(action)
                    state = np.array(state, dtype=np.float32).flatten()
                    
                    frame = env.render()
                    if frame is not None:
                        img = Image.fromarray(frame)
                        img.thumbnail((600, 400)) # Compress/scale to save bandwidth
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=80)
                        jpeg_bytes = buf.getvalue()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                               
                    if terminated or truncated:
                        break
                    await asyncio.sleep(0.04) # roughly 25 frames per second
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in testing stream: {e}")
        finally:
            if env is not None:
                env.close()
                
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
