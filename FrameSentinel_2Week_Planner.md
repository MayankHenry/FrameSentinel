# FrameSentinel — 2-Week Build Planner
**Real-Time Multi-Camera Safety & Anomaly Monitoring Platform**

Scope for these 14 days: a working end-to-end pipeline (RTSP/video → YOLO → tracker → zone rules → alerts → dashboard), containerized, with tests on the rule engine and a documented edge-vs-cloud decision. Kafka, Jetson deployment, and multi-tenant auth polish are explicitly **out of scope** — call them "future scope" in the README.

**Ground rules for the two weeks:**
- Commit at least once per working session with a meaningful message (`feat:`, `fix:`, `test:`, `docs:`). This project is solo-owned by design — no sole-contributor flag risk, but a thin commit history still reads badly to reviewers. Aim for 20+ substantive commits by Day 14, not 3 giant ones.
- Every day ends with something *runnable*, even if incomplete. No days where you only "read docs."
- Keep a `DECISIONS.md` from Day 1 — one line per architectural choice + why. This is your interview-prep cheat sheet later.

---

## Week 1 — Core Pipeline (Detection → Tracking → Rules → Alerts)

### Day 1 — Project Skeleton + Video Ingestion
- Repo init: `frame-sentinel/` with `inference-service/`, `api-service/`, `frontend/`, `docker/`, `docs/`
- Set up Python env for inference service (Python 3.10/3.11, `ultralytics`, `opencv-python`, `redis`)
- Ingestion module: read from a **video file or webcam** first (not live RTSP yet) — simulate a camera with a looped sample video (e.g., a warehouse/construction stock clip)
- Frame sampling logic: pull every Nth frame (configurable FPS, start at 5–10 fps)
- Deliverable: script that reads a video source and displays sampled frames with OpenCV
- Commit: `feat: video ingestion + frame sampling`

### Day 2 — YOLO Detection Integration
- Load YOLOv8n (nano, for speed) via `ultralytics`; run inference on sampled frames
- Draw bounding boxes + class labels + confidence on output frames
- Log inference latency per frame (this feeds your metrics story later)
- Test on 2–3 different videos (people walking, people near objects) to sanity-check detection quality
- Deliverable: annotated video/frame output with detection overlays
- Commit: `feat: YOLOv8 detection integration + latency logging`

### Day 3 — Tracker Integration (ByteTrack)
- Integrate ByteTrack (via `ultralytics` built-in tracking mode, or standalone) to assign persistent IDs across frames
- Store per-track state: track_id, bbox history, first_seen, last_seen
- Deliverable: video with consistent ID labels per person across frames (not re-numbering every frame)
- Commit: `feat: ByteTrack integration for persistent object IDs`

### Day 4 — Zone Definition + Zone-Intrusion Rule
- Data model: `zones` table (camera_id, polygon_coords as JSON, rule_type)
- For now, hardcode 1–2 test polygons per video (JSON file is fine before DB exists)
- Rule logic: is bbox-center (or bottom-center) inside polygon? (point-in-polygon check)
- On intrusion: emit an internal event object `{camera_id, zone_id, track_id, rule, confidence, timestamp}`
- **Start the rule engine as a standalone, pure-logic module** — no I/O, so it's trivially unit-testable
- Deliverable: script prints/logs an event when a tracked person enters the test polygon
- Commit: `feat: polygon zone-intrusion rule engine`

### Day 5 — Loitering + Alert Deduplication Logic
- Loitering rule: track_id stationary (small centroid movement) inside a zone for >X seconds → trigger
- Dedup logic: don't re-fire the same (track_id, zone_id, rule) alert within a cooldown window (e.g., 30–60s) — this is the "don't fire 50 alerts for one event" requirement, and it's a strong interview talking point
- Write this dedup logic so it's swappable (in-memory dict for now, Redis-backed later)
- Deliverable: loitering scenario triggers exactly one alert, not a stream of duplicates
- Commit: `feat: loitering detection + alert deduplication`

### Day 6 — Rule Engine Unit Tests
- This is pure logic — no excuse to skip tests. Use `pytest`.
- Test cases: point inside/outside/on-edge of polygon; loitering timer edge cases; dedup window boundaries; multiple simultaneous tracks in different zones
- Deliverable: `tests/test_rule_engine.py` with 10+ passing tests, plus a coverage report
- Commit: `test: rule engine unit tests (zone, loitering, dedup)`

### Day 7 — Redis Pub/Sub + Clip Extraction
- Stand up Redis (Docker container is fine locally)
- Inference service publishes alert events to a Redis channel instead of just logging
- On trigger, use FFmpeg to extract a short clip (~5–10s around the event) from the buffered frames/video and save locally (S3/R2 comes later)
- Deliverable: alert event published to Redis + corresponding clip file saved to disk with a matching filename/ID
- Commit: `feat: Redis pub/sub for alerts + FFmpeg clip extraction`

**End of Week 1 checkpoint:** you should be able to run one script against a test video and see: detections → tracked IDs → zone/loitering rules firing → deduped alerts → clips saved → events on Redis. Everything after this is "wrap this in services and put it on a screen."

---

## Week 2 — Services, Dashboard, Deployment, Polish

### Day 8 — FastAPI Backend: DB + Core Endpoints
- Postgres schema: `cameras`, `zones`, `alerts`, `detections` (optional), `users`
- FastAPI service: Redis subscriber consumes alert events → writes to `alerts` table
- REST endpoints: CRUD for cameras/zones, GET alerts (paginated, filterable by camera/date)
- Deliverable: alert fired by inference service lands as a row in Postgres, retrievable via API
- Commit: `feat: FastAPI backend with Postgres persistence + Redis consumer`

### Day 9 — Auth + RBAC
- JWT auth (login endpoint, token issuance)
- Roles: `site_manager` (full access, can edit zones) vs `viewer` (read-only alerts)
- Protect zone-editing endpoints behind `site_manager` role
- Deliverable: login flow works, role-gated endpoint returns 403 for viewer role
- Commit: `feat: JWT auth + role-based access control`

### Day 10 — WebSocket Alert Push + Frontend Skeleton
- FastAPI WebSocket endpoint: pushes new alerts to connected clients in real time
- React + Tailwind skeleton: login page, dashboard shell, alert feed component (static first, then wire to WebSocket)
- Deliverable: triggering a test alert shows up live in the React alert feed without refresh
- Commit: `feat: WebSocket alert streaming + React dashboard skeleton`

### Day 11 — Live Video Tiles + Clip Playback
- Package saved clips for browser playback (HLS packaging via FFmpeg, or simplest: serve as MP4 and use `<video>` tag — don't over-engineer HLS if MP4 playback meets the bar)
- Alert feed item → "View Clip" opens the corresponding clip
- If time allows: live tile showing current camera feed (HLS.js) — otherwise document as a stretch goal, not a blocker
- Deliverable: clicking an alert in the dashboard plays back the triggering clip
- Commit: `feat: clip playback from alert feed`

### Day 12 — Drag-to-Draw Zone UI (the frontend-design flex)
- Canvas/SVG-based polygon editor overlaid on a camera reference frame
- Site manager draws a polygon, assigns rule type (zone-intrusion / loitering / PPE), saves to backend
- This is your strongest visual/portfolio screenshot — spend real time on the interaction polish, not just function
- Deliverable: a zone drawn in the UI is persisted and actually enforced by the rule engine on the next run
- Commit: `feat: drag-to-draw zone configuration UI`

### Day 13 — Docker Compose + Config for Confidence Thresholds
- `docker-compose.yml`: inference service, API service, Redis, Postgres, (frontend can stay on Vercel/local dev)
- Per-rule configurable confidence thresholds (env var or DB-backed config, not hardcoded)
- Basic logging/metrics: inference latency, dropped frame count, exposed via a `/metrics` endpoint or simple log aggregation
- Sanity-test "N concurrent streams" — even if N=2–3 locally, document what breaks first (this is your answer to the "100 cameras" interview question)
- Deliverable: `docker-compose up` brings up the full stack from a clean machine
- Commit: `feat: Docker Compose orchestration + configurable thresholds`

### Day 14 — Edge/Cloud Decision, README, Resume Polish
- Be honest in `docs/ARCHITECTURE.md`: did you actually export to ONNX/TensorRT and benchmark, or are you documenting it as a decision you'd make with hardware access? Either is defensible — dishonesty in the interview is not.
- If time: quick ONNX export + a CPU vs GPU latency comparison table (even a rough one is real signal)
- Write final README: problem statement, architecture diagram, tech stack, setup instructions, screenshots/GIF of the zone editor + live alert feed, known limitations, future scope (Kafka, Jetson, multi-tenant)
- Finalize resume bullets against what you actually built (not the aspirational version)
- Commit: `docs: architecture writeup + README + demo screenshots`

---

## Interview Prep Checklist (map to what you built, not theory)
- [ ] Frame drops / camera reconnect handling — what did you actually implement vs. what's future scope?
- [ ] Why YOLOv8n specifically — did you compare against YOLOv5/v8s or another detector? Even one comparison run counts.
- [ ] Alert dedup — walk through your cooldown-window logic with the actual code.
- [ ] 100-camera scaling — name the real bottleneck (likely: single inference process pinned to one GPU/CPU; discuss horizontal scaling of inference workers behind Redis).

## Explicitly Out of Scope (say so, don't fake it)
- Kafka (mention Redis→Kafka migration path in "future scope")
- Jetson Nano physical deployment (document ONNX/TensorRT export path; deploy target = cloud GPU unless you actually have the hardware)
- Multi-tenant org structure beyond two roles
- HLS live-tile streaming, if Day 11 runs long — MP4 playback is an acceptable substitute
