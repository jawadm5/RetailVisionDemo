"""YOLO inference service — CPU dev mode (x86 / Intel, no CUDA).

Identical ZMQ wire protocol to service.py.
Replaces TRT engine with ultralytics YOLO .pt model running on CPU.
For local development only — do NOT deploy to production.

Architecture (unchanged from service.py):
  IEP2 × N  ──PUSH──►  PULL (yolo_input.sock)
                            ↓ batch collector
                        YOLO .pt CPU inference
                            ↓ per-camera routing
  IEP2 × N  ◄──PUSH──  PUSH(yolo_output_{camera_id}.sock)
"""

import asyncio
import json
import logging
import os
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq
import zmq.asyncio
from prometheus_client import Counter, Gauge, Histogram, start_http_server

log = logging.getLogger("yolo_service_dev")

# ── Prometheus metrics (job "detector", scraped on :9400) ──────────────────────
# Named "detector_*" not "yolo_*" because the real detector is RT-DETR-x
# (rtdetr-x.pt); dev uses a lighter YOLO11n stand-in. The actual model in use is
# exposed via the detector_info{model=...} gauge. Defined once at import time.
DETECTOR_INFER = Histogram(
    "detector_inference_seconds",
    "Detection latency per batch",
)
DETECTOR_BATCH = Histogram(
    "detector_batch_size",
    "Frames per inference batch",
    buckets=[1, 2, 4, 8, 16, 32],
)
DETECTOR_FRAMES = Counter("detector_frames_total", "Frames processed")
DETECTOR_DETECTIONS = Counter("detector_detections_total", "Person detections returned")
# Which detector model is actually loaded (RT-DETR-x in prod, YOLO11n in dev).
# Value is always 1; the information lives in the label. Set at startup.
DETECTOR_INFO = Gauge("detector_info", "Loaded detector model (value always 1)", ["model"])

# ── Device selection (CPU / GPU) shared with the dev pipeline ──────────────────
# The dev screen's CPU/GPU toggle writes the desired device to Redis key
# `inference:device`; this service resolves it against the hardware it can
# actually see and applies it live. It publishes what the machine supports to
# `inference:capability:detector` so EEP can answer "does this machine have a GPU?".
_REDIS_URL          = os.environ.get("SERVER_REDIS_URL") or os.environ.get("REDIS_URL", "redis://redis:6379/0")
_DEVICE_KEY         = "inference:device"               # desired: cpu|cuda|xpu|gpu|auto
_CAP_KEY            = "inference:capability:detector"
_INITIAL_DEVICE     = os.environ.get("INFERENCE_DEVICE", "cpu")

# Mutable holder read by the inference loop; updated by the watcher thread.
_state = {"device": "cpu"}


def _detect_caps() -> dict:
    """Which GPU backends this machine/container can actually use."""
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        xpu  = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    except Exception:
        cuda = xpu = False
    return {"cuda": cuda, "xpu": xpu}


def _resolve_device(requested: str, caps: dict) -> str:
    """Map a requested device against real capability. Falls back to cpu."""
    requested = (requested or "cpu").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if caps["cuda"] else "cpu"
    if requested == "xpu":
        return "xpu" if caps["xpu"] else "cpu"
    # "gpu" / "auto": prefer CUDA, then Intel XPU
    if caps["cuda"]:
        return "cuda"
    if caps["xpu"]:
        return "xpu"
    return "cpu"


def _device_watcher() -> None:
    """Background thread: publish capability + apply the requested device live."""
    try:
        import redis
        r = redis.Redis.from_url(_REDIS_URL)
    except Exception as exc:
        log.warning("Device watcher disabled (redis unavailable): %s", exc)
        return
    caps = _detect_caps()
    log.info("Detector GPU capability: cuda=%s xpu=%s", caps["cuda"], caps["xpu"])
    while True:
        try:
            requested = r.get(_DEVICE_KEY)
            requested = requested.decode() if requested else _INITIAL_DEVICE
            resolved = _resolve_device(requested, caps)
            if resolved != _state["device"]:
                log.info("Inference device → %s (requested=%s)", resolved, requested)
                _state["device"] = resolved
            r.set(_CAP_KEY, json.dumps({
                "role": "detector", "cuda": caps["cuda"], "xpu": caps["xpu"],
                "device": _state["device"],
            }), ex=15)
        except Exception as exc:
            log.warning("Device watcher tick failed: %s", exc)
        time.sleep(2)

# ── Configuration ─────────────────────────────────────────────────────────────
YOLO_INPUT_SOCK       = os.environ.get("YOLO_INPUT_SOCK",        "ipc:///tmp/sockets/yolo_input.sock")
YOLO_HEALTH_UNIX_SOCK = os.environ.get("YOLO_HEALTH_SOCK",       "unix:///tmp/sockets/yolo_health.sock")
YOLO_HEALTH_TCP_ADDR  = os.environ.get("YOLO_HEALTH_TCP_ADDR",   "[::]:50052")

# Dev detector model. Loader class is chosen by filename:
# yolo11*.pt → YOLO; rtdetr-*.pt → RTDETR; *.engine → TensorRT.
DETECTOR_MODEL        = os.environ.get("DETECTOR_MODEL",         "yolo11n.pt")
YOLO_CONF_THRESHOLD   = float(os.environ.get("YOLO_CONF",        "0.5"))
YOLO_IOU_THRESHOLD    = float(os.environ.get("YOLO_IOU",         "0.45"))
# Smaller defaults on CPU — increase via GPU overlay.
MAX_BATCH_SIZE        = int(os.environ.get("YOLO_MAX_BATCH_SIZE", "4"))
BATCH_TIMEOUT_MS      = float(os.environ.get("YOLO_BATCH_TIMEOUT_MS", "500"))
# Idle gap that ends batch collection. Small on purpose: it only needs to
# cover a burst already in flight, not to wait for new work.
BATCH_DRAIN_GRACE_MS   = float(os.environ.get("YOLO_BATCH_DRAIN_GRACE_MS", "2"))
# TRT lazy-compile: set EXPORT_TRT=true to auto-export .pt → .engine on first GPU run.
EXPORT_TRT            = os.environ.get("EXPORT_TRT", "false").lower() == "true"

# ── Per-camera result sockets ─────────────────────────────────────────────────
_result_sockets: dict[str, zmq.asyncio.Socket] = {}


def _result_sock_addr(camera_id: str) -> str:
    return f"ipc:///tmp/sockets/yolo_output_{camera_id}.sock"


def _get_result_socket(ctx: zmq.asyncio.Context, camera_id: str) -> zmq.asyncio.Socket:
    if camera_id not in _result_sockets:
        sock = ctx.socket(zmq.PUSH)
        sock.connect(_result_sock_addr(camera_id))
        _result_sockets[camera_id] = sock
        log.info("Opened result channel for camera %s", camera_id)
    return _result_sockets[camera_id]


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(model_file: str):
    """Load an ultralytics detector. Picks loader by filename.

    TRT lazy-compile: if EXPORT_TRT=true and running on GPU, converts the .pt
    to a TensorRT .engine on first run (~5 min), then loads the engine on every
    subsequent start. Requires the image to be built with INSTALL_TRT=true.
    """
    device = _state.get("device", "cpu")
    base   = os.path.basename(model_file).lower()

    # ── TensorRT lazy-compile ─────────────────────────────────────────────────
    if EXPORT_TRT and device != "cpu" and model_file.endswith(".pt"):
        engine_file = os.path.splitext(model_file)[0] + ".engine"
        if os.path.exists(engine_file):
            log.info("TensorRT engine found — loading %s", engine_file)
            model_file = engine_file
            base       = os.path.basename(model_file).lower()
        else:
            try:
                import tensorrt  # type: ignore[import-untyped]  # noqa: F401
                from ultralytics import RTDETR, YOLO  # type: ignore[import-untyped]  # noqa: F811
                log.info(
                    "Exporting %s → TensorRT engine (first-run, ~5 min)…", model_file
                )
                _Loader = RTDETR if base.startswith("rtdetr") else YOLO
                _Loader(model_file).export(format="engine", half=True, imgsz=640, device=0)
                model_file = engine_file
                base       = os.path.basename(model_file).lower()
                log.info("TensorRT engine ready: %s", engine_file)
            except ImportError:
                log.warning(
                    "tensorrt not installed — skipping TRT export. "
                    "Rebuild with INSTALL_TRT=true in docker-compose.gpu.yml to enable."
                )

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model not found: {model_file}. "
            "Ensure weights are baked into the image by Dockerfile.dev."
        )

    if base.startswith("rtdetr"):
        from ultralytics import RTDETR as _Model
        family = "RT-DETR"
    else:
        from ultralytics import YOLO as _Model
        family = "YOLO"
    log.info("Loading %s model: %s", family, model_file)
    return _Model(model_file)


def _warmup(model) -> None:
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    model([blank], verbose=False, device=_state["device"])
    log.info("Warmup complete on device=%s", _state["device"])


# ── Inference ─────────────────────────────────────────────────────────────────

def _decode_frame(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return np.zeros((640, 640, 3), dtype=np.uint8)
    return frame


def _get_frame(item: dict) -> np.ndarray:
    """Resolve one request's frame: read from the shared /dev/shm ``frame_path``
    when present (the frame-path transport — avoids an IEP2-side re-encode), else
    decode the JPEG ``frame`` bytes carried in the message. Falls back to bytes if
    the path is unreadable, so a mount glitch degrades instead of dropping frames."""
    path = item.get("frame_path")
    if path:
        frame = cv2.imread(path)
        if frame is not None:
            return frame
        log.warning("frame_path unreadable (%s) — falling back to bytes", path)
    data = item.get("frame")
    if data:
        return _decode_frame(data)
    return np.zeros((640, 640, 3), dtype=np.uint8)


def _infer_batch(model, batch_items: list[dict]) -> list[dict]:
    """Batch inference — R4 filters to class 0 (person) only. Identical to service.py."""
    DETECTOR_BATCH.observe(len(batch_items))
    frames = [_get_frame(item) for item in batch_items]
    with DETECTOR_INFER.time():
        results = model(
            frames,
            verbose=False,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            device=_state["device"],
            half=_state["device"] != "cpu",
        )
    DETECTOR_FRAMES.inc(len(frames))
    responses = []
    for item, result in zip(batch_items, results):
        detections = []
        if result.boxes is not None:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs     = result.boxes.conf.cpu().numpy()
            xyxys     = result.boxes.xyxy.cpu().numpy()
            for cls_id, conf, xyxy in zip(class_ids, confs, xyxys):
                if cls_id != 0:   # R4: person only
                    continue
                detections.append({
                    "bbox_xyxy":  [float(x) for x in xyxy],
                    "confidence": float(conf),
                })
        DETECTOR_DETECTIONS.inc(len(detections))
        responses.append({
            "request_id":   item["request_id"],
            "camera_id":    item["camera_id"],
            "timestamp_ms": item["timestamp_ms"],
            "detections":   detections,
        })
    return responses


# ── Batch collector ────────────────────────────────────────────────────────────

async def _collect_batch(pull_sock: zmq.asyncio.Socket) -> list[dict]:
    raw = await pull_sock.recv()
    batch = [msgpack.unpackb(raw, raw=False)]

    # Collection ends on whichever comes first: the batch is full, the queue has
    # been idle for BATCH_DRAIN_GRACE_MS, or BATCH_TIMEOUT_MS total has elapsed.
    #
    # The idle-gap condition is the important one. Callers here are strictly
    # request/response: IEP2 sends a crop (or frame) and then blocks awaiting
    # that specific reply, so once the in-flight messages are drained NOTHING
    # further can arrive until we answer. Waiting for MAX_BATCH_SIZE therefore
    # burned the entire BATCH_TIMEOUT_MS on every batch — measured at ~50 ms x
    # ~545 batches per 60 s window, about 27 s of a 35 s ReID phase spent
    # waiting for messages that could not come. Genuine batching still happens
    # whenever several cameras are active, because their requests are actually
    # concurrent and are already queued when we drain.
    deadline = time.monotonic() + BATCH_TIMEOUT_MS / 1000.0
    grace = BATCH_DRAIN_GRACE_MS / 1000.0
    while len(batch) < MAX_BATCH_SIZE:
        remaining = min(grace, deadline - time.monotonic())
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(pull_sock.recv(), timeout=remaining)
            batch.append(msgpack.unpackb(raw, raw=False))
        except asyncio.TimeoutError:
            break

    return batch


# ── Inference loop ─────────────────────────────────────────────────────────────

async def _inference_loop(
    model,
    pull_sock: zmq.asyncio.Socket,
    ctx: zmq.asyncio.Context,
) -> None:
    loop = asyncio.get_running_loop()
    while True:
        batch = await _collect_batch(pull_sock)
        responses = await loop.run_in_executor(None, _infer_batch, model, batch)
        for resp in responses:
            sock = _get_result_socket(ctx, resp["camera_id"])
            await sock.send(msgpack.packb(resp, use_bin_type=True))
        log.debug("Batch processed  size=%d", len(batch))


# ── Health server ──────────────────────────────────────────────────────────────

async def _run_health_server(health_servicer) -> None:
    import grpc.aio
    from grpc_health.v1 import health_pb2_grpc

    server = grpc.aio.server()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    server.add_insecure_port(YOLO_HEALTH_UNIX_SOCK)
    server.add_insecure_port(YOLO_HEALTH_TCP_ADDR)
    await server.start()
    log.info("Health server: unix=%s  tcp=%s", YOLO_HEALTH_UNIX_SOCK, YOLO_HEALTH_TCP_ADDR)
    await server.wait_for_termination()


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from grpc_health.v1 import health, health_pb2

    health_servicer = health.HealthServicer()
    health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)

    asyncio.create_task(_run_health_server(health_servicer))
    await asyncio.sleep(0)

    # Prometheus /metrics on :9400 (own background thread). Scraped as job "detector".
    start_http_server(9400)
    DETECTOR_INFO.labels(model=os.path.basename(DETECTOR_MODEL)).set(1)
    log.info("Prometheus metrics server started on :9400 (model=%s)", DETECTOR_MODEL)

    # Resolve the initial device synchronously (so warmup uses it), then start
    # the watcher thread that keeps it in sync with the dev screen's toggle.
    _state["device"] = _resolve_device(_INITIAL_DEVICE, _detect_caps())
    threading.Thread(target=_device_watcher, name="device-watcher", daemon=True).start()

    log.info("Loading detector  file=%s  conf=%.2f  device=%s", DETECTOR_MODEL, YOLO_CONF_THRESHOLD, _state["device"])
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(None, _load_model, DETECTOR_MODEL)
    await loop.run_in_executor(None, _warmup, model)

    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    log.info("Detector service (dev/CPU) SERVING  model=%s  conf=%.2f", DETECTOR_MODEL, YOLO_CONF_THRESHOLD)

    os.makedirs("/tmp/sockets", exist_ok=True)
    ctx = zmq.asyncio.Context.instance()
    pull_sock = ctx.socket(zmq.PULL)
    pull_sock.bind(YOLO_INPUT_SOCK)
    log.info("Bound input socket: %s", YOLO_INPUT_SOCK)

    await _inference_loop(model, pull_sock, ctx)


if __name__ == "__main__":
    asyncio.run(main())
