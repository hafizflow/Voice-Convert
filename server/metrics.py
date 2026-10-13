from prometheus_client import Counter, Gauge, Histogram

_MS_BUCKETS = (0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320)

ACTIVE = Gauge("vc_active_sessions", "Active conversion sessions")
INFERENCE_MS = Histogram("vc_inference_ms", "Per-block converter time in ms", buckets=_MS_BUCKETS)
SERVER_LATENCY_MS = Histogram(
    "vc_server_latency_ms", "Frame received to converted frame handed to publisher, ms", buckets=_MS_BUCKETS
)
FRAME_ERRORS = Counter("vc_frame_errors", "Frames that failed processing (silence was sent)")
