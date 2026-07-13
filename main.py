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
from dataset import class_counts, save_sample
from inference import predict
from preprocess import frame_to_sample
from stream import Camera, CameraStreamTrack
from train import main as train_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEARCH_SPIN_DUTY = 0.3


@dataclass
class AppState:
    mode: str = "label"
    state: str = "stop"
    capturing: bool = False
    camera: Camera = field(default_factory=Camera)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model: object | None = None
    mean: object | None = None
    components: object | None = None
    latest_prediction: dict | None = None
    ws_clients: set[WebSocket] = field(default_factory=set)
    peer_connections: set[RTCPeerConnection] = field(default_factory=set)


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
    if state.latest_prediction is not None:
        pred = state.latest_prediction
        parts.append(f"pred={pred['class_name']} ({pred['confidence']:.0%})")
    return " | ".join(parts)


def apply_search_stub() -> None:
    if state.state == "search":
        logger.info("Search mode active (CCW spin stub, duty=%.1f)", SEARCH_SPIN_DUTY)


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
            }
            state.latest_prediction = message

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

    sample_task = asyncio.create_task(sample_loop())
    predict_task = asyncio.create_task(predict_loop())
    yield
    sample_task.cancel()
    predict_task.cancel()
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


@app.post("/state")
def set_state(body: StateRequest) -> dict:
    if body.state not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid state: {body.state}")
    state.state = body.state
    apply_search_stub()
    return {"ok": True, "state": state.state}


@app.post("/mode")
def set_mode(body: ModeRequest) -> dict:
    if body.mode not in ("label", "infer"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {body.mode}")
    if body.mode == "infer" and not artifacts_exist():
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Run `uv run python train.py` first.",
        )
    state.mode = body.mode
    if body.mode == "infer":
        state.capturing = False
        loaded = load_artifacts(state.device)
        if loaded is not None:
            state.model, state.mean, state.components, _ = loaded
    return {"ok": True, "mode": state.mode}


@app.post("/capture")
def set_capture(body: CaptureRequest) -> dict:
    if body.capturing and state.mode != "label":
        raise HTTPException(status_code=400, detail="Capture only works in label mode")
    state.capturing = body.capturing
    return {"ok": True, "capturing": state.capturing}


@app.post("/train")
def trigger_train() -> dict:
    train_main()
    loaded = load_artifacts(state.device)
    if loaded is not None:
        state.model, state.mean, state.components, _ = loaded
    return {"ok": True, "message": "Training complete"}


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
