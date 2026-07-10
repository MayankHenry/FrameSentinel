<div align="center">

# 🛡️ FrameSentinel

### Real-Time Multi-Camera Safety & Anomaly Monitoring Platform

**Enterprise-grade CV surveillance, without the enterprise price tag.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Redis](https://img.shields.io/badge/Redis-PubSub-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

[![Status](https://img.shields.io/badge/status-in%20development-yellow?style=flat-square)]()
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)]()
[![Build Progress](https://img.shields.io/badge/build-Day%201%2F14-orange?style=flat-square)]()

</div>

---

## 📸 What is this?

Small warehouses, retail stores, and construction sites can't afford enterprise CV surveillance platforms like **Verkada**. **FrameSentinel** ingests multiple RTSP camera streams, runs real-time object detection + configurable anomaly rules — zone intrusion, PPE-not-worn, loitering, fall detection — and pushes live alerts to a dashboard with clip playback. Deployable on a cheap edge device (Jetson Nano / Raspberry Pi + Coral) or a cloud GPU instance.

> **Why it exists:** applied CV + real-time systems + edge deployment is a hiring lane that's distinct from GenAI work — industrial, robotics, and safety-tech companies specifically screen for it, and it's one of the few project categories where hardware-awareness actually shows.

<br>

<div align="center">

| 🏭 Warehouses | 🏗️ Construction Sites | 🏬 Retail Stores |
|:---:|:---:|:---:|
| Restricted-zone intrusion | PPE compliance (hard hat / vest) | Loitering detection |
| Forklift-zone safety | Fall detection | After-hours monitoring |

</div>

---

## 🎯 Core Features

- 🎥 **Multi-camera ingestion** — RTSP streams sampled at configurable FPS, decoupled from source frame rate
- 🧠 **Real-time detection + tracking** — YOLOv8/v9 detection, ByteTrack for persistent IDs across frames
- 📐 **Configurable zone rule engine** — drag-to-draw polygons, pure-logic and fully unit-tested, decoupled from the detection model
- 🚨 **Smart alerting** — deduplication so a single 10-second event doesn't spam 50 alerts; per-rule confidence thresholds
- 🎬 **Live dashboard + clip playback** — WebSocket-pushed alerts, thumbnail + clip link, no page refresh
- 🔐 **Auth + RBAC** — JWT-based, site manager vs. viewer roles
- 🐳 **One-command deploy** — Docker Compose spins up inference service, API, Redis, and Postgres together
- 📊 **Inference observability** — latency and dropped-frame logging built in from Day 1, not bolted on later

---

## 🏗️ System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐
│  RTSP Camera │───▶│ Frame Sampler │───▶│  YOLOv8/v9  │───▶│ ByteTrack Tracker │
│   Stream(s)  │    │  (target FPS) │    │  Detection  │    │  (persistent IDs) │
└─────────────┘    └──────────────┘    └─────────────┘    └────────┬─────────┘
                                                                     │
                                                                     ▼
                                                        ┌────────────────────────┐
                                                        │   Zone / Rule Engine    │
                                                        │  (intrusion, loitering, │
                                                        │   PPE, fall detection)  │
                                                        └───────────┬────────────┘
                                                                    │ on trigger
                                                     ┌──────────────┴──────────────┐
                                                     ▼                             ▼
                                          ┌────────────────────┐        ┌───────────────────┐
                                          │  Redis Pub/Sub      │        │  FFmpeg Clip        │
                                          │  (alert event)       │        │  Extraction         │
                                          └─────────┬──────────┘        └─────────┬─────────┘
                                                    │                              │
                                                    ▼                              ▼
                                        ┌────────────────────┐        ┌────────────────────┐
                                        │  FastAPI Consumer   │        │  S3 / R2 Blob Store │
                                        │  → PostgreSQL        │        │  (stored clips)     │
                                        └─────────┬──────────┘        └─────────────────────┘
                                                  │
                                                  ▼ WebSocket push
                                    ┌───────────────────────────┐
                                    │   React + Tailwind Dashboard  │
                                    │   Live alert feed · clip playback │
                                    │   Drag-to-draw zone editor    │
                                    └───────────────────────────┘
```

**Why two services, not one?** The API service (FastAPI) and the inference service (Python, GPU-bound) are deliberately separate processes. Inference is CPU/GPU-heavy and scales horizontally per camera load; the API is I/O-bound and scales differently. Coupling them would mean scaling both together for no reason — this separation is itself a system-design decision, documented in [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## 🧰 Tech Stack

<table>
<tr>
<td valign="top" width="50%">

**Frontend**
- React + TailwindCSS
- HLS.js / WebRTC for live tiles
- Canvas/SVG drag-to-draw zone editor

**Backend**
- FastAPI (API + auth)
- Separate Python inference service
- JWT auth, role-based access control

**CV / ML**
- YOLOv8 / YOLOv9 (Ultralytics)
- ByteTrack (multi-object tracking)
- MediaPipe Pose (fall detection, optional)

</td>
<td valign="top" width="50%">

**Streaming**
- OpenCV / GStreamer (RTSP ingestion)
- FFmpeg (clip extraction)
- HLS packaging for browser playback

**Data**
- PostgreSQL (cameras, zones, alerts, detections)
- Redis Pub/Sub (alert event bus)
- S3 / Cloudflare R2 (clip storage)

**Deploy**
- Docker Compose (local + cloud GPU)
- ONNX / TensorRT export path (edge target)
- Frontend on Vercel / Netlify

</td>
</tr>
</table>

---

## 🚀 Quickstart

```bash
git clone https://github.com/<your-username>/FrameSentinel.git
cd FrameSentinel/inference-service

pip install -r requirements.txt

# Drop a test video into sample_videos/, then:
python src/ingest.py --source sample_videos/your_video.mp4 --target-fps 8
```

> 📦 Full `docker-compose up` for the entire stack (inference + API + Redis + Postgres) lands in Week 2 — see [Roadmap](#-roadmap) below.

---

## 📅 Roadmap

<details>
<summary><strong>Week 1 — Core Pipeline</strong> (click to expand)</summary>

- [x] Day 1 — Project skeleton + video ingestion & frame sampling
- [ ] Day 2 — YOLOv8 detection integration
- [ ] Day 3 — ByteTrack persistent-ID tracking
- [ ] Day 4 — Zone definition + zone-intrusion rule
- [ ] Day 5 — Loitering detection + alert deduplication
- [ ] Day 6 — Rule engine unit tests
- [ ] Day 7 — Redis pub/sub + FFmpeg clip extraction

</details>

<details>
<summary><strong>Week 2 — Services, Dashboard, Deployment</strong></summary>

- [ ] Day 8 — FastAPI backend + Postgres persistence
- [ ] Day 9 — JWT auth + RBAC
- [ ] Day 10 — WebSocket alert push + React dashboard skeleton
- [ ] Day 11 — Live video tiles + clip playback
- [ ] Day 12 — Drag-to-draw zone configuration UI
- [ ] Day 13 — Docker Compose + configurable thresholds
- [ ] Day 14 — Edge/cloud benchmarking + final docs

</details>

<details>
<summary><strong>Explicitly out of scope (for now)</strong></summary>

- Kafka (Redis is sufficient at this scale — Kafka noted as future scope)
- Physical Jetson Nano deployment (ONNX/TensorRT export path documented; cloud GPU used unless hardware is available)
- Multi-tenant org structure beyond two roles
- Live HLS tile streaming (MP4 clip playback is the fallback)

</details>

---

## 🗃️ Database Schema (core tables)

| Table | Purpose |
|---|---|
| `cameras` | Registered camera streams, connection info |
| `zones` | Polygon coordinates + rule type per camera |
| `alerts` | Camera, zone, rule, confidence, clip URL, timestamp |
| `detections` | Raw detection log (optional — eval/replay) |

---

## 🧪 Testing

The rule engine is pure logic (zone containment, loitering timers, dedup windows) with no I/O — built to be exhaustively unit tested:

```bash
cd inference-service
pytest tests/ -v --cov=src
```

---

## 💡 Interview-Ready Talking Points

- **Frame drops / camera reconnects** — how the ingestion pipeline handles a dropped RTSP stream without crashing
- **Why YOLOv8** — actual benchmark comparisons, not just "it's popular"
- **Alert spam prevention** — the deduplication/cooldown-window logic behind a single flickering detection
- **Scaling to 100 cameras** — where the real bottleneck is, and how inference workers scale horizontally behind Redis

Full reasoning behind every architectural choice is logged in [`docs/DECISIONS.md`](docs/DECISIONS.md) as it's built — not written retroactively.

---

<div align="center">

**Built by Team 85 · GLA University**

*A portfolio project demonstrating applied computer vision, real-time systems design, and edge-deployment awareness.*

</div>
