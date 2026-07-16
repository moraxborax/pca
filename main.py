import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from artifacts import artifacts_exist, load_artifacts
from config import (
    CLASSES,
    DATASET_SPLITS,
    DEFAULT_TRAIN_EPOCHS,
    MAX_TRAIN_EPOCHS,
    MIN_TRAIN_EPOCHS,
    SAMPLE_RATE_HZ,
)
from control import apply_action, init_board, stop as motor_stop
from dataset import class_counts, clear_dataset, ensure_class_dirs, migrate_legacy_layout, save_sample
from inference import predict
from preprocess import frame_to_sample
from stream import Camera, CameraStreamTrack
from train import main as train_main
from training_progress import fail as fail_training_progress
from training_progress import snapshot as training_progress_snapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AppState:
    mode: str = "label"
    state: str = "stop"
    capturing: bool = False
    dataset_split: str = "train"
    infer_paused: bool = False
    last_driven_action: str | None = None
    camera: Camera = field(default_factory=Camera)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model: object | None = None
    mean: object | None = None
    components: object | None = None
    artifact_config: dict | None = None
    latest_prediction: dict | None = None
    ws_clients: set[WebSocket] = field(default_factory=set)
    peer_connections: set[RTCPeerConnection] = field(default_factory=set)
    board: object | None = None


state = AppState()


class StateRequest(BaseModel):
    state: str


class ModeRequest(BaseModel):
    mode: str


class CaptureRequest(BaseModel):
    capturing: bool


class DatasetSplitRequest(BaseModel):
    split: str


class OfferRequest(BaseModel):
    sdp: str
    type: str


class TrainRequest(BaseModel):
    epochs: int = DEFAULT_TRAIN_EPOCHS


def overlay_text() -> str:
    parts = [f"mode={state.mode}", f"state={state.state}"]
    if state.capturing:
        parts.append(f"REC:{state.dataset_split}")
    if state.mode == "infer" and state.infer_paused:
        parts.append("PAUSED")
    if state.latest_prediction is not None:
        pred = state.latest_prediction
        label = pred["class_name"]
        if pred.get("forced_search"):
            label = f"{label}*"
        parts.append(f"pred={label} ({pred['confidence']:.0%})")
        if "residual" in pred:
            parts.append(f"r={pred['residual']:.2f}")
    return " | ".join(parts)


def drive_motors(action: str, *, force: bool = False) -> None:
    if not force and action == state.last_driven_action:
        return
    try:
        apply_action(state.board, action)
        state.last_driven_action = action
    except Exception:
        logger.exception("Motor command failed for action=%s", action)


def try_init_motors() -> None:
    try:
        state.board = init_board()
        logger.info("Motor board ready")
    except Exception:
        state.board = None
        logger.warning("Motor board unavailable — running without motors", exc_info=True)


def apply_loaded_artifacts(loaded: tuple) -> None:
    state.model, state.mean, state.components, state.artifact_config = loaded
    threshold = state.artifact_config.get("residual_threshold")
    logger.info(
        "Loaded model artifacts (%s components, residual_threshold=%s)",
        state.artifact_config.get("n_components"),
        threshold,
    )


async def sample_loop() -> None:
    interval = 1.0 / SAMPLE_RATE_HZ
    while True:
        try:
            await asyncio.sleep(interval)
            if state.mode != "label" or not state.capturing:
                continue
            frame = await asyncio.to_thread(state.camera.get_latest_frame)
            if frame is None:
                continue
            sample = frame_to_sample(frame)
            await asyncio.to_thread(
                save_sample, state.state, sample, state.dataset_split
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("sample_loop error (will retry)")


async def predict_loop() -> None:
    interval = 1.0 / SAMPLE_RATE_HZ
    while True:
        try:
            await asyncio.sleep(interval)
            if state.mode != "infer" or state.model is None:
                continue

            frame = await asyncio.to_thread(state.camera.get_latest_frame)
            if frame is None:
                continue

            residual_threshold = None
            if isinstance(getattr(state, "artifact_config", None), dict):
                residual_threshold = state.artifact_config.get("residual_threshold")

            prediction = await asyncio.to_thread(
                predict,
                state.model,
                state.mean,
                state.components,
                frame,
                state.device,
                residual_threshold,
            )
            message = {
                "class_name": prediction.class_name,
                "class_index": prediction.class_index,
                "confidence": prediction.confidence,
                "commanded_state": state.state,
                "driving": not state.infer_paused,
                "paused": state.infer_paused,
                "residual": prediction.residual,
                "forced_search": prediction.forced_search,
            }
            state.latest_prediction = message

            if state.infer_paused:
                await asyncio.to_thread(drive_motors, "stop")
            else:
                await asyncio.to_thread(drive_motors, prediction.class_name)

            dead_clients: list[WebSocket] = []
            for ws in state.ws_clients:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead_clients.append(ws)
            for ws in dead_clients:
                state.ws_clients.discard(ws)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("predict_loop error (will retry)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    migrate_legacy_layout()
    ensure_class_dirs()

    loaded = load_artifacts(state.device)
    if loaded is not None:
        apply_loaded_artifacts(loaded)
    else:
        logger.info("No model artifacts found — train after collecting data")

    await asyncio.to_thread(try_init_motors)

    sample_task = asyncio.create_task(sample_loop())
    predict_task = asyncio.create_task(predict_loop())
    yield
    sample_task.cancel()
    predict_task.cancel()
    if state.board is not None:
        try:
            await asyncio.to_thread(motor_stop, state.board)
        except Exception:
            logger.exception("Failed to stop motors on shutdown")
    for pc in list(state.peer_connections):
        await pc.close()
    state.camera.release()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/dataset/stats")
def get_dataset_stats() -> dict:
    counts = class_counts()
    return {
        "counts": counts,
        "active_split": state.dataset_split,
        "train_total": sum(counts["train"].values()),
        "test_total": sum(counts["test"].values()),
    }


@app.post("/dataset/split")
def set_dataset_split(body: DatasetSplitRequest) -> dict:
    if body.split not in DATASET_SPLITS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid split: {body.split}. Use train or test.",
        )
    state.dataset_split = body.split
    return {"ok": True, "split": state.dataset_split}


@app.delete("/dataset")
def delete_dataset(split: str | None = None) -> dict:
    state.capturing = False
    if split is not None and split not in (*DATASET_SPLITS, "all"):
        raise HTTPException(
            status_code=400,
            detail="split must be train, test, or all",
        )
    target = None if split in (None, "all") else split
    counts = clear_dataset(target)
    return {"ok": True, "counts": counts}


@app.post("/state")
def set_state(body: StateRequest) -> dict:
    if body.state not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid state: {body.state}")
    state.state = body.state

    if state.mode == "infer":
        # Q/stop = emergency stop; any other key resumes ML driving
        if body.state == "stop":
            state.infer_paused = True
            drive_motors("stop", force=True)
        else:
            state.infer_paused = False
        return {
            "ok": True,
            "state": state.state,
            "motors": state.board is not None,
            "infer_paused": state.infer_paused,
            "driving": "ml" if not state.infer_paused else "paused",
        }

    # Capture / train: keyboard drives motors directly
    drive_motors(body.state)
    return {"ok": True, "state": state.state, "motors": state.board is not None}


@app.post("/mode")
def set_mode(body: ModeRequest) -> dict:
    if body.mode not in ("label", "train", "infer"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {body.mode}")
    if body.mode == "infer" and not artifacts_exist():
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Use the Train tab first.",
        )
    prev = state.mode
    state.mode = body.mode
    if body.mode != "label":
        state.capturing = False
    if body.mode == "train":
        state.state = "stop"
        state.infer_paused = True
        drive_motors("stop", force=True)
    if body.mode == "infer":
        loaded = load_artifacts(state.device)
        if loaded is not None:
            apply_loaded_artifacts(loaded)
        state.infer_paused = False
        state.state = "stop"
        drive_motors("stop", force=True)
    if prev == "infer" and body.mode != "infer":
        drive_motors("stop", force=True)
    return {"ok": True, "mode": state.mode}


@app.post("/capture")
def set_capture(body: CaptureRequest) -> dict:
    if body.capturing and state.mode != "label":
        raise HTTPException(status_code=400, detail="Capture only works in label mode")
    state.capturing = body.capturing
    return {"ok": True, "capturing": state.capturing}


@app.post("/train")
def trigger_train(body: TrainRequest = TrainRequest()) -> dict:
    epochs = body.epochs
    if epochs < MIN_TRAIN_EPOCHS or epochs > MAX_TRAIN_EPOCHS:
        raise HTTPException(
            status_code=400,
            detail=f"epochs must be between {MIN_TRAIN_EPOCHS} and {MAX_TRAIN_EPOCHS}",
        )
    try:
        history = train_main(epochs=epochs)
    except ValueError as exc:
        fail_training_progress(str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        fail_training_progress(str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    loaded = load_artifacts(state.device)
    if loaded is not None:
        apply_loaded_artifacts(loaded)
    return {
        "ok": True,
        "message": "Training complete — switch to Infer to try it",
        "history": history,
    }


@app.get("/train/progress")
def get_train_progress() -> dict:
    return training_progress_snapshot()


@app.get("/train/metrics")
def get_train_metrics() -> dict:
    from viz import load_training_history

    history = load_training_history()
    if history is None:
        raise HTTPException(status_code=404, detail="No training history yet. Train first.")
    return history


@app.get("/pca/viz")
def get_pca_viz() -> dict:
    from viz import load_pca_viz

    viz = load_pca_viz()
    if viz is None:
        raise HTTPException(status_code=404, detail="No PCA artifacts yet. Train first.")
    return viz


@app.post("/offer")
async def offer(body: OfferRequest) -> dict:
    offer_desc = RTCSessionDescription(sdp=body.sdp, type=body.type)
    pc = RTCPeerConnection()
    state.peer_connections.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            state.peer_connections.discard(pc)

    track = CameraStreamTrack(state.camera, overlay_text=overlay_text)
    pc.addTrack(track)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket) -> None:
    await websocket.accept()
    state.ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.ws_clients.discard(websocket)
