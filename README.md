# FrameSentinel

Real-Time Multi-Camera Safety & Anomaly Monitoring Platform.
See `docs/DECISIONS.md` for architecture reasoning as it's built.

**Status:** Day 1 of 14 — video ingestion + frame sampling.

## Day 1 — Ingestion Smoke Test

```bash
cd inference-service
pip install -r requirements.txt
python src/ingest.py --source sample_videos/your_test_video.mp4 --target-fps 8
```

Drop any test video (warehouse/construction/pedestrian footage works well)
into `inference-service/sample_videos/` — that folder is gitignored so raw
video files never get committed.

No test video handy yet? Point `--source` at your webcam:
```bash
python src/ingest.py --source 0 --target-fps 8 --display
```

### What this proves today
- Video source (file or webcam) opens reliably
- Frames are sampled at a fixed target FPS independent of the source's native FPS
- Throughput is logged (sampled frame count, last source frame index, elapsed time) —
  this same logging pattern gets extended in Day 2+ into latency/dropped-frame metrics

### Repo layout
```
frame-sentinel/
├── inference-service/   # YOLO + tracker + rule engine (Python)
│   ├── src/
│   ├── sample_videos/   # gitignored — drop test clips here
│   └── requirements.txt
├── api-service/         # FastAPI backend (Day 8+)
├── frontend/            # React + Tailwind dashboard (Day 10+)
├── docker/              # docker-compose.yml (Day 13)
└── docs/
    └── DECISIONS.md
```
