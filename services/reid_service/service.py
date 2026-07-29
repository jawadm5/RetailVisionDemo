"""ReID embedding service — resnet50_msmt17 (2048-dim).

Single GPU process that batches person crops from all IEP2 workers and runs
one TRT inference call per batch. Returns L2-normalised 2048-dim float32 embeddings.

Architecture mirrors yolo-service:
  IEP2 × N  ──PUSH──►  PULL (reid_input.sock)
                            ↓ batch collector
                        TRT inference (executor)
                            ↓ per-camera routing + L2 normalise
  IEP2 × N  ◄──PUSH──  PUSH(reid_output_{camera_id}.sock)

R1: TRT engine only — .pt at runtime is banned.
R3: Preprocessing owned by this service (decode/resize/normalise).
R5: L2 normalisation before return.
R6: Assert 2048-dim output.
R7: NOT_SERVING during load, SERVING after warmup.
R8: ipc:// sockets only.
"""

import asyncio
import logging
import os
import time

import cv2
import msgpack
import numpy as np
import zmq
import zmq.asyncio
from prometheus_client import Counter, Histogram, start_http_server

log = logging.getLogger("reid_service")

# ── Configuration ─────────────────────────────────────────────────────────────
REID_INPUT_SOCK       = os.environ.get("REID_INPUT_SOCK",       "ipc:///tmp/sockets/reid_input.sock")
REID_HEALTH_UNIX_SOCK = os.environ.get("REID_HEALTH_SOCK",      "unix:///tmp/sockets/reid_health.sock")
REID_HEALTH_TCP_ADDR  = os.environ.get("REID_HEALTH_TCP_ADDR",  "[::]:50053")

REID_MODEL_PATH  = os.environ.get("REID_MODEL_PATH",  "resnet50_msmt17.engine")
MAX_BATCH_SIZE    = int(os.environ.get("REID_MAX_BATCH_SIZE",    "64"))
BATCH_TIMEOUT_MS  = float(os.environ.get("REID_BATCH_TIMEOUT_MS", "50"))
# Idle gap that ends batch collection. Small on purpose: it only needs to
# cover a burst already in flight, not to wait for new work.
BATCH_DRAIN_GRACE_MS   = float(os.environ.get("REID_BATCH_DRAIN_GRACE_MS", "2"))
EMBEDDING_DIM     = 2048
REID_METRICS_PORT = int(os.environ.get("REID_METRICS_PORT", "9401"))

# ── Prometheus metrics (scraped on :9401, job "reid") ─────────────────────────
# model_version label carries metrics that reflect model output quality;
# infra/throughput metrics do not. Only additive instrumentation — no behaviour.
_MODEL_VERSION = os.environ.get("MODEL_VERSION", "production")

REID_CROPS = Counter(
    "reid_crops_processed_total",
    "Total person crops processed by the ReID embedding service",
)
REID_ERRORS = Counter(
    "reid_errors_total",
    "Crops that failed to preprocess or caused an inference error",
)
REID_INFERENCE = Histogram(
    "reid_inference_seconds",
    "Wall-clock time for one TRT ReID batch inference call",
    ["model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
REID_BATCH_SIZE = Histogram(
    "reid_batch_size",
    "Number of crops per TRT ReID inference batch — GPU utilisation proxy",
    buckets=[1, 2, 4, 8, 16, 32, 48, 64, 96, 128],
)
# ML signal: distribution of raw (pre-L2) embedding norms. A collapse toward 0
# indicates degenerate embeddings before any downstream metric degrades.
REID_EMBEDDING_NORM = Histogram(
    "reid_embedding_norm",
    "L2 norm of raw ResNet50 embeddings before normalisation (2048-dim)",
    ["model_version"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0],
)

# ImageNet normalisation constants (R3).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Per-camera result sockets ─────────────────────────────────────────────────
_result_sockets: dict[str, zmq.asyncio.Socket] = {}


def _result_sock_addr(camera_id: str) -> str:
    return f"ipc:///tmp/sockets/reid_output_{camera_id}.sock"


def _get_result_socket(ctx: zmq.asyncio.Context, camera_id: str) -> zmq.asyncio.Socket:
    if camera_id not in _result_sockets:
        sock = ctx.socket(zmq.PUSH)
        sock.connect(_result_sock_addr(camera_id))
        _result_sockets[camera_id] = sock
        log.info("Opened result channel for camera %s", camera_id)
    return _result_sockets[camera_id]


# ── Preprocessing (R3) ────────────────────────────────────────────────────────

def _preprocess_crop(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG bytes → CHW float16 tensor normalised to ImageNet stats."""
    arr  = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if crop is None:
        crop = np.zeros((256, 128, 3), dtype=np.uint8)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (128, 256))                    # W×H = 128×256
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - _MEAN) / _STD
    return crop.transpose(2, 0, 1).astype(np.float16)     # HWC → CHW, fp16


# ── TRT engine loading and inference ─────────────────────────────────────────

def _load_engine(engine_path: str):
    """Load TRT engine via tensorrt Python API."""
    if not os.path.exists(engine_path):
        raise FileNotFoundError(
            f"TRT engine not found: {engine_path}. "
            "Rebuild the image — export_reid_trt.py runs at build time."
        )
    import tensorrt as trt
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialise TRT engine: {engine_path}")
    log.info("TRT engine loaded: %s", engine_path)
    return engine


def _warmup(engine) -> None:
    """One forward pass with blank batch to warm up CUDA kernels."""
    blank = np.zeros((1, 3, 256, 128), dtype=np.float16)
    _infer_batch(engine, blank)
    log.info("ReID TRT warmup complete.")


def _infer_batch(engine, batch: np.ndarray) -> np.ndarray:
    """Run TRT inference.

    batch: float16 [B, 3, 256, 128] contiguous host array.
    Returns: float32 [B, 2048].
    """
    import pycuda.driver as cuda

    context = engine.create_execution_context()
    context.set_input_shape("input", batch.shape)

    batch_c = np.ascontiguousarray(batch.astype(np.float16))
    output  = np.empty((batch.shape[0], EMBEDDING_DIM), dtype=np.float16)

    d_input  = cuda.mem_alloc(batch_c.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)
    stream   = cuda.Stream()

    cuda.memcpy_htod_async(d_input, batch_c, stream)
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)],
        stream_handle=stream.handle,
    )
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    d_input.free()
    d_output.free()

    return output.astype(np.float32)


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    """R5: L2 normalise a (2048,) embedding in-place."""
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return emb
    return emb / norm


def _infer_and_pack(engine, batch_items: list[dict]) -> list[dict]:
    """Preprocess, infer, L2-normalise, pack responses."""
    # Preprocess all crops to float16 tensors.
    tensors = [_preprocess_crop(item["crop"]) for item in batch_items]
    batch   = np.stack(tensors, axis=0)              # [B, 3, 256, 128] fp16

    t0 = time.monotonic()
    embeddings = _infer_batch(engine, batch)          # [B, 2048] fp32
    REID_INFERENCE.labels(model_version=_MODEL_VERSION).observe(time.monotonic() - t0)
    REID_BATCH_SIZE.observe(len(batch_items))
    REID_CROPS.inc(len(batch_items))

    responses = []
    for item, emb in zip(batch_items, embeddings):
        # R6: assert dimension is correct.
        assert emb.shape == (EMBEDDING_DIM,), f"Expected ({EMBEDDING_DIM},), got {emb.shape}"
        # ML signal: record raw norm before L2 normalisation.
        REID_EMBEDDING_NORM.labels(model_version=_MODEL_VERSION).observe(float(np.linalg.norm(emb)))
        emb = _l2_normalize(emb)                      # R5
        responses.append({
            "request_id":   item["request_id"],
            "camera_id":    item["camera_id"],
            "track_id":     item["track_id"],
            "timestamp_ms": item["timestamp_ms"],
            "embedding":    emb.astype(np.float32).tobytes(),  # R2: raw bytes, 2048 bytes
        })
    return responses


# ── Batch collector ────────────────────────────────────────────────────────────

async def _collect_batch(pull_sock: zmq.asyncio.Socket) -> list[dict]:
    """Block until first crop arrives, then collect until MAX_BATCH_SIZE or BATCH_TIMEOUT_MS."""
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
    engine,
    pull_sock: zmq.asyncio.Socket,
    ctx: zmq.asyncio.Context,
) -> None:
    loop = asyncio.get_running_loop()
    while True:
        batch = await _collect_batch(pull_sock)
        responses = await loop.run_in_executor(None, _infer_and_pack, engine, batch)
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
    server.add_insecure_port(REID_HEALTH_UNIX_SOCK)
    server.add_insecure_port(REID_HEALTH_TCP_ADDR)
    await server.start()
    log.info("Health server: unix=%s  tcp=%s", REID_HEALTH_UNIX_SOCK, REID_HEALTH_TCP_ADDR)
    await server.wait_for_termination()


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    import pycuda.autoinit  # noqa: F401 — initialises CUDA context on import

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Prometheus metrics HTTP server — started before engine load so Prometheus
    # sees the target as UP while the model is still initialising.
    start_http_server(REID_METRICS_PORT)
    log.info("Prometheus metrics server started on :%d", REID_METRICS_PORT)

    from grpc_health.v1 import health, health_pb2

    health_servicer = health.HealthServicer()
    health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)  # R7

    asyncio.create_task(_run_health_server(health_servicer))
    await asyncio.sleep(0)

    log.info("Loading TRT engine: %s  max_batch=%d  timeout_ms=%.0f",
             REID_MODEL_PATH, MAX_BATCH_SIZE, BATCH_TIMEOUT_MS)
    loop = asyncio.get_running_loop()
    engine = await loop.run_in_executor(None, _load_engine, REID_MODEL_PATH)
    await loop.run_in_executor(None, _warmup, engine)

    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)  # R7
    log.info("ReID service SERVING  model=%s  dim=%d", REID_MODEL_PATH, EMBEDDING_DIM)

    os.makedirs("/tmp/sockets", exist_ok=True)
    ctx = zmq.asyncio.Context.instance()
    pull_sock = ctx.socket(zmq.PULL)
    pull_sock.bind(REID_INPUT_SOCK)
    log.info("Bound input socket: %s", REID_INPUT_SOCK)

    await _inference_loop(engine, pull_sock, ctx)


if __name__ == "__main__":
    asyncio.run(main())
