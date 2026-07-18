# FrameSentinel — Architecture Decisions Log

One line per decision + why. Keep this updated daily — this is your interview
cheat sheet later, not documentation for its own sake.

| Date | Decision | Why |
|------|----------|-----|
| Day 1 | `cv2.VideoCapture` for ingestion (not raw GStreamer pipeline yet) | Handles file/webcam/RTSP via the same API — no code change needed when real RTSP cameras replace test videos. Revisit if RTSP reconnect handling proves insufficient. |
| Day 1 | Sample via `.grab()` skip + `.retrieve()` on target frames, not decode-every-frame-then-drop | `.grab()` is far cheaper than full decode; matters once running multiple camera streams concurrently. |
| Day 1 | Target FPS configurable, decoupled from source FPS | Inference load is bounded by target FPS regardless of camera's native frame rate — this is the actual lever for scaling to more concurrent streams later. |
| Day 1 | `sampled_frames` per logged second doesn't equal target FPS | Ingestion runs at max decode speed for offline files (batch throughput), not throttled to real-time playback. Only live RTSP streams are naturally paced by the camera. No code change needed — just a logging nuance worth explaining if asked. |
| Day 2 | YOLOv8n (nano) as the default model, not s/m/l | Smallest/fastest variant — right default until an actual latency-vs-accuracy benchmark justifies a bigger model. Benchmark comparison is a Week 2/Day 14 task, not assumed now. |
| Day 2 | `Detector` class only exposes `DetectionResult` (dataclass), never raw Ultralytics objects | Keeps the rule engine (Day 4+) decoupled from the detection library. Swapping YOLOv8 for another detector later should be a config change, not a rewrite across the codebase. |
| Day 2 | Per-frame inference latency logged from Day 2, not added later | Same metric feeds Day 13's dropped-frame/latency observability story — cheaper to log it from the start than retrofit. |
| Day 3 | Used Ultralytics' built-in `model.track()` (ByteTrack) instead of a standalone ByteTrack implementation | Same underlying algorithm, far less integration surface area. Trade-off: less "I built the tracker from scratch" story, more "I understand tracker integration + its failure modes" story — the latter is more defensible under questioning. |
| Day 3 | Centroid defined as bbox bottom-center, not box center | Bottom-center approximates where a person is actually standing (feet position), which is what matters for zone-containment checks in Day 4 — box center would be roughly torso height, less accurate for "is this person inside the polygon." |
| Day 3 | `tracker_cfg` restricted to a whitelist (`bytetrack.yaml`/`botsort.yaml`), not an arbitrary path | If this pipeline is ever wrapped behind an API, an unrestricted config path is a file-read/config-injection vector. Cheap to lock down now, easy to forget later. |
| Day 3 | RTSP credentials masked in all logs (`mask_credentials`) | Camera URLs often carry `user:pass@host` — never want that landing in a log file or terminal history, even in a local dev script. |
| Day 3 | Per-frame tracking exceptions caught and skipped (return empty result), not raised | A single corrupt/flaky frame shouldn't kill a multi-hour camera session. Bad frames are logged as warnings; the pipeline keeps running. |