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
from config import CLASSES, SAMPLE_RATE_HZ
from control import apply_action, init_board, stop as motor_stop
from dataset import class_counts, clear_dataset, save_sample
from inference import predict
from preprocess import frame_to_sample
from stream import Camera, CameraStreamTrack
from train import main as train_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AppState:
    mode: str = "label"
    state: str = "stop"
    capturing: bool = False
    infer_paused: bool = False
    last_driven_action: str | None = None
    camera: Camera = field(default_factory=Camera)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model: object | None = None
    mean: object | None = None
    components: object | None = None
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


class OfferRequest(BaseModel):
    sdp: str
    type: str


def overlay_text() -> str:
    parts = [f"mode={state.mode}", f"state={state.state}"]
    if state.capturing:
        parts.append("REC")
    if state.mode == "infer" and state.infer_paused:
        parts.append("PAUSED")
    if state.latest_prediction is not None:
        pred = state.latest_prediction
        parts.append(f"pred={pred['class_name']} ({pred['confidence']:.0%})")
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
            await asyncio.to_thread(save_sample, state.state, sample)
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

            prediction = await asyncio.to_thread(
                predict, state.model, state.mean, state.components, frame, state.device
            )
            message = {
                "class_name": prediction.class_name,
                "class_index": prediction.class_index,
                "confidence": prediction.confidence,
                "commanded_state": state.state,
                "driving": not state.infer_paused,
                "paused": state.infer_paused,
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
    loaded = load_artifacts(state.device)
    if loaded is not None:
        state.model, state.mean, state.components, config = loaded
        logger.info("Loaded model artifacts (%d components)", config["n_components"])
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
    return {"counts": class_counts()}


@app.delete("/dataset")
def delete_dataset() -> dict:
    state.capturing = False
    counts = clear_dataset()
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
            state.model, state.mean, state.components, _ = loaded
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
def trigger_train() -> dict:
    try:
        train_main()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    loaded = load_artifacts(state.device)
    if loaded is not None:
        state.model, state.mean, state.components, _ = loaded
    return {"ok": True, "message": "Training complete — switch to Infer to try it"}


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
