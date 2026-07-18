"""
FrameSentinel — ByteTrack Persistent-ID Tracking
Day 3

Adds object tracking on top of Day 2's detections: each detected object gets
a persistent track_id that survives across frames, instead of YOLO
re-numbering detections from scratch every frame. This is required before
any zone/loitering rule can work — "has this person been in this zone for
10 seconds" is meaningless without a stable identity to measure against.

Uses Ultralytics' built-in ByteTrack integration (model.track(), not
model.predict()) rather than wiring a standalone ByteTrack implementation —
same underlying algorithm, far less integration surface area for a resume
project. Documented as a deliberate scope decision in docs/DECISIONS.md.

Usage:
    python src/track.py --source sample_videos/sample.mp4 --target-fps 8 --display
    python src/track.py --source rtsp://user:pass@host:554/stream --target-fps 8

Error handling & security notes (Day 3 hardening pass):
    - Model path is validated before load; local .pt files are checked to
      exist, remote/hub names are allowed through to Ultralytics as-is.
    - `--tracker-cfg` is restricted to a whitelist of known-safe Ultralytics
      configs (bytetrack.yaml / botsort.yaml) rather than accepting an
      arbitrary path. If this pipeline is ever wrapped behind an API, an
      unrestricted config path is a file-read/config-injection vector —
      worth keeping locked down even in a local dev script.
    - RTSP credentials (user:pass@host) are masked in all log/print output —
      never log a raw camera URL if it contains a password.
    - Per-frame failures (corrupt frame, tracker exception) are caught and
      skipped with a warning rather than crashing the whole run — a bad
      frame from a flaky camera shouldn't kill a multi-hour session.
    - Resources (VideoWriter, display window, model) are released in a
      `finally` block so a Ctrl+C or mid-run exception doesn't leave file
      handles or GUI windows open.
"""

import argparse
import logging
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from ingest import VideoIngestor, Frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("frame-sentinel.track")

# Only these tracker configs are accepted — do not widen this to an arbitrary
# path without also validating it, especially once this runs behind an API.
ALLOWED_TRACKER_CONFIGS = {"bytetrack.yaml", "botsort.yaml"}

_CREDENTIAL_RE = re.compile(r"(://)([^:@/]+):([^@/]+)@")


def mask_credentials(source: str) -> str:
    """Redacts user:pass in an RTSP/HTTP URL before it ever hits a log line."""
    if not isinstance(source, str):
        return str(source)
    return _CREDENTIAL_RE.sub(r"\1***:***@", source)


@dataclass
class TrackedObject:
    """A single tracked object in one frame."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    centroid: Tuple[float, float]            # bottom-center point, used for zone checks


@dataclass
class TrackingResult:
    frame: Frame
    tracks: List[TrackedObject] = field(default_factory=list)
    inference_ms: float = 0.0


class TrackHistory:
    """
    Keeps a short rolling history of each track_id's centroid + last-seen time.

    Intentionally minimal — just enough to support Day 4's zone check and
    Day 5's loitering timer. This is NOT the rule engine; it only tracks
    "where has this ID been," the rule engine decides what that means.
    """

    def __init__(self, max_history: int = 60, stale_after_s: float = 30.0):
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        self.max_history = max_history
        self.stale_after_s = stale_after_s
        self._centroids: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self.first_seen: Dict[int, float] = {}
        self.last_seen: Dict[int, float] = {}

    def update(self, tracks: List[TrackedObject], timestamp: float) -> None:
        active_ids = set()
        for t in tracks:
            active_ids.add(t.track_id)
            self._centroids[t.track_id].append(t.centroid)
            if t.track_id not in self.first_seen:
                self.first_seen[t.track_id] = timestamp
            self.last_seen[t.track_id] = timestamp
        self._prune_stale(timestamp)

    def _prune_stale(self, timestamp: float) -> None:
        """Drop tracks not seen recently so memory doesn't grow unbounded over a long stream."""
        stale = [tid for tid, last in self.last_seen.items()
                 if timestamp - last > self.stale_after_s]
        for tid in stale:
            self._centroids.pop(tid, None)
            self.first_seen.pop(tid, None)
            self.last_seen.pop(tid, None)

    def duration_active(self, track_id: int, now: float) -> float:
        """How long (seconds) this track_id has been continuously observed."""
        return now - self.first_seen.get(track_id, now)

    def centroid_history(self, track_id: int) -> Deque[Tuple[float, float]]:
        return self._centroids.get(track_id, deque())

    def active_track_count(self) -> int:
        return len(self.last_seen)


class Tracker:
    """
    Wraps YOLO's built-in ByteTrack mode. Like Detector in Day 2, this only
    ever hands out TrackedObject / TrackingResult — never raw Ultralytics
    objects — so the rule engine built in Day 4+ stays decoupled from the
    tracking library choice.
    """

    def __init__(self, model_path: str = "models/yolov8n.pt", conf_threshold: float = 0.35,
                 classes: Optional[List[int]] = None, tracker_cfg: str = "bytetrack.yaml"):
        if not (0.0 < conf_threshold <= 1.0):
            raise ValueError(f"conf_threshold must be in (0, 1], got {conf_threshold}")
        if tracker_cfg not in ALLOWED_TRACKER_CONFIGS:
            raise ValueError(
                f"tracker_cfg '{tracker_cfg}' not allowed. "
                f"Must be one of: {sorted(ALLOWED_TRACKER_CONFIGS)}"
            )

        # Only enforce "must exist" for local weight files. Ultralytics hub
        # names (e.g. "yolov8n.pt" with no path separators) are allowed
        # through as-is since Ultralytics auto-downloads those.
        model_file = Path(model_path)
        looks_like_local_path = model_file.suffix == ".pt" and (
            model_file.is_absolute() or len(model_file.parts) > 1
        )
        if looks_like_local_path and not model_file.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                f"Run Day 2's detect.py first (it auto-downloads yolov8n.pt), "
                f"or pass a valid path/hub name."
            )

        try:
            self.model = YOLO(model_path)
        except Exception as exc:  # Ultralytics raises assorted exception types on bad weights
            raise RuntimeError(f"Failed to load YOLO model from '{model_path}': {exc}") from exc

        self.conf_threshold = conf_threshold
        self.classes = classes
        self.tracker_cfg = tracker_cfg

    def track(self, frame: Frame) -> TrackingResult:
        if frame is None or frame.image is None or frame.image.size == 0:
            log.warning("Skipping empty/corrupt frame (index=%s)",
                        getattr(frame, "index", "unknown"))
            return TrackingResult(frame=frame, tracks=[], inference_ms=0.0)

        start = time.perf_counter()
        try:
            results = self.model.track(
                frame.image,
                conf=self.conf_threshold,
                classes=self.classes,
                tracker=self.tracker_cfg,
                persist=True,  # keep tracker state alive across calls on this model instance
                verbose=False,
            )
        except Exception as exc:
            # A single bad frame (e.g. transient decode artifact) shouldn't
            # kill a long-running camera session — log and return empty.
            log.error("Tracking inference failed on frame %s: %s",
                      getattr(frame, "index", "unknown"), exc)
            return TrackingResult(frame=frame, tracks=[], inference_ms=0.0)

        inference_ms = (time.perf_counter() - start) * 1000

        tracks: List[TrackedObject] = []
        if not results:
            return TrackingResult(frame=frame, tracks=tracks, inference_ms=inference_ms)

        result = results[0]

        # boxes.id is None on frames where the tracker hasn't assigned IDs yet
        # (e.g. very first frame, or a frame with zero detections)
        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                try:
                    track_id = int(box.id[0])
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                except (TypeError, ValueError, IndexError) as exc:
                    log.warning("Skipping malformed detection box: %s", exc)
                    continue

                class_name = self.model.names.get(class_id, f"class_{class_id}") \
                    if isinstance(self.model.names, dict) else str(class_id)
                centroid = ((x1 + x2) / 2, y2)  # bottom-center: proxy for "where someone is standing"

                tracks.append(TrackedObject(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    centroid=centroid,
                ))

        return TrackingResult(frame=frame, tracks=tracks, inference_ms=inference_ms)


def draw_tracks(image, result: TrackingResult):
    """Draws bounding boxes with persistent track IDs (color-coded per ID)."""
    if image is None:
        return image
    annotated = image.copy()
    for t in result.tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        # deterministic color per track_id so the same ID looks the same across frames
        color = (
            (t.track_id * 37) % 255,
            (t.track_id * 91) % 255,
            (t.track_id * 143) % 255,
        )
        label = f"ID {t.track_id} | {t.class_name} {t.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cx, cy = (int(v) for v in t.centroid)
        cv2.circle(annotated, (cx, cy), 4, color, -1)

    cv2.putText(annotated, f"inference: {result.inference_ms:.1f} ms", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    return annotated


def parse_classes(raw: Optional[str]) -> Optional[List[int]]:
    """Parses '--classes 0,2,3' safely, rejecting non-integer input instead of crashing deep in YOLO."""
    if not raw:
        return None
    try:
        return [int(c.strip()) for c in raw.split(",") if c.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--classes must be a comma-separated list of integers, got '{raw}'"
        ) from exc


def resolve_source(raw_source: str):
    """
    Validates the --source argument before it's handed to OpenCV.

    - Pure digits -> webcam index (int)
    - Starts with a known stream scheme -> passed through as-is
    - Otherwise treated as a local file path and checked to exist, so a typo
      fails fast with a clear message instead of an opaque OpenCV error.
    """
    if raw_source.isdigit():
        return int(raw_source)

    stream_schemes = ("rtsp://", "rtsps://", "http://", "https://")
    if raw_source.lower().startswith(stream_schemes):
        return raw_source

    path = Path(raw_source)
    if not path.exists():
        raise FileNotFoundError(f"Video source not found: '{raw_source}'")
    if not path.is_file():
        raise ValueError(f"Video source is not a file: '{raw_source}'")
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="FrameSentinel — Day 3 tracking smoke test")
    parser.add_argument("--source", required=True,
                         help="Video file path, webcam index, or RTSP/HTTP URL")
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--classes", type=str, default="0",
                         help="Comma-separated COCO class IDs, default '0' = person only")
    parser.add_argument("--tracker-cfg", type=str, default="bytetrack.yaml",
                         choices=sorted(ALLOWED_TRACKER_CONFIGS))
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", type=str, default=None)
    args = parser.parse_args()

    if args.target_fps <= 0:
        parser.error("--target-fps must be positive")
    if not (0.0 < args.conf <= 1.0):
        parser.error("--conf must be in (0, 1]")

    try:
        source = resolve_source(args.source)
        classes = parse_classes(args.classes)
    except (FileNotFoundError, ValueError, argparse.ArgumentTypeError) as exc:
        log.error(str(exc))
        return 1

    try:
        tracker = Tracker(model_path=args.model, conf_threshold=args.conf,
                           classes=classes, tracker_cfg=args.tracker_cfg)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        log.error("Could not initialize tracker: %s", exc)
        return 1

    history = TrackHistory()
    writer = None
    frame_count = 0
    unique_ids = set()
    display_active = False

    log.info("Starting tracking on source=%s target_fps=%s conf=%s classes=%s",
              mask_credentials(str(args.source)), args.target_fps, args.conf, classes)

    try:
        with VideoIngestor(source, target_fps=args.target_fps) as ingestor:
            for frame in ingestor.frames():
                result = tracker.track(frame)
                history.update(result.tracks, frame.timestamp)
                frame_count += 1
                for t in result.tracks:
                    unique_ids.add(t.track_id)

                annotated = draw_tracks(frame.image, result)
                if annotated is None:
                    continue

                if args.save_video:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                            args.target_fps, (w, h),
                        )
                        if not writer.isOpened():
                            log.error("Could not open VideoWriter for '%s' — disabling video save",
                                      args.save_video)
                            writer = None
                    if writer is not None:
                        writer.write(annotated)

                if args.display:
                    display_active = True
                    cv2.imshow("FrameSentinel — Day 3 (ByteTrack IDs)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("Quit requested by user ('q' pressed)")
                        break

                if frame_count % 20 == 0:
                    log.info("frames=%d active_tracks_this_frame=%d unique_ids_so_far=%d",
                              frame_count, len(result.tracks), len(unique_ids))

    except RuntimeError as exc:
        # Raised by VideoIngestor.open() if the source can't be opened at all
        log.error("Ingestion error: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C) — shutting down cleanly")
    finally:
        if writer is not None:
            writer.release()
        if display_active:
            cv2.destroyAllWindows()

    log.info("Done. total_frames=%d total_unique_ids=%d active_at_end=%d",
              frame_count, len(unique_ids), history.active_track_count())

    if unique_ids and frame_count > 0 and len(unique_ids) > frame_count:
        log.warning("unique_ids (%d) exceeds total frames processed (%d) — likely heavy "
                    "ID-switching. Worth a DECISIONS.md note and a --conf tweak.",
                    len(unique_ids), frame_count)

    return 0


if __name__ == "__main__":
    sys.exit(main())