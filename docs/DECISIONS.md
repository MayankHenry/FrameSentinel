# FrameSentinel — Architecture Decisions Log

One line per decision + why. Keep this updated daily — this is your interview
cheat sheet later, not documentation for its own sake.

| Date | Decision | Why |
|------|----------|-----|
| Day 1 | `cv2.VideoCapture` for ingestion (not raw GStreamer pipeline yet) | Handles file/webcam/RTSP via the same API — no code change needed when real RTSP cameras replace test videos. Revisit if RTSP reconnect handling proves insufficient. |
| Day 1 | Sample via `.grab()` skip + `.retrieve()` on target frames, not decode-every-frame-then-drop | `.grab()` is far cheaper than full decode; matters once running multiple camera streams concurrently. |
| Day 1 | Target FPS configurable, decoupled from source FPS | Inference load is bounded by target FPS regardless of camera's native frame rate — this is the actual lever for scaling to more concurrent streams later. |
