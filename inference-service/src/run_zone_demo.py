"""
FrameSentinel — Day 4 Runner: Tracking + Zone-Intrusion Rules
Day 4

Wires together:
    ingest.py     (Day 1) — frame sampling
    track.py      (Day 3) — YOLOv8 + ByteTrack, persistent track IDs
    rule_engine.py(Day 4) — pure-logic zone-intrusion detection

Draws the configured zone polygon(s) alongside tracked boxes, and prints an
alert line whenever a track enters a zone (edge-triggered — see
docs/DECISIONS.md for why this doesn't fire every frame someone remains inside).

Usage:
    python src/run_zone_demo.py --source ../sample_videos/sample.mp4 \
        --zones-config ../configs/zones.example.json --camera-id cam_1 --display

If you don't have a zones config yet, generate one first:
    python src/zone_picker.py --source ../sample_videos/sample.mp4 \
        --camera-id cam_1 --output ../configs/zones.example.json
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from ingest import VideoIngestor
from track import Tracker, resolve_source, parse_classes, mask_credentials
from rule_engine import RuleEngine, TrackSnapshot, load_zones_from_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("frame-sentinel.run_zone_demo")


def draw_zones(image, zones, camera_id: str):
    annotated = image.copy()
    for zone in zones:
        if zone.camera_id != camera_id:
            continue
        color = (255, 0, 255) if zone.rule_type == "zone_intrusion" else (0, 140, 255)
        pts = [(int(x), int(y)) for x, y in zone.polygon]
        for i in range(len(pts)):
            cv2.line(annotated, pts[i], pts[(i + 1) % len(pts)], color, 2)
        label_pos = pts[0]
        label = f"{zone.name} ({zone.rule_type})"
        cv2.putText(annotated, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def draw_tracks(image, tracks):
    annotated = image.copy()
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        color = ((t.track_id * 37) % 255, (t.track_id * 91) % 255, (t.track_id * 143) % 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"ID {t.track_id}", (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cx, cy = (int(v) for v in t.centroid)
        cv2.circle(annotated, (cx, cy), 4, color, -1)
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description="FrameSentinel — Day 4 zone-intrusion demo")
    parser.add_argument("--source", required=True)
    parser.add_argument("--zones-config", required=True, help="Path to zones JSON config")
    parser.add_argument("--camera-id", required=True, help="camera_id to evaluate zones for")
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--classes", type=str, default="0", help="default '0' = person only")
    parser.add_argument("--dedup-cooldown", type=float, default=30.0,
                         help="Seconds before the same (track, zone, rule) can alert again")
    parser.add_argument("--debug-centroids", action="store_true",
                         help="Print every track's centroid each frame — use this to "
                              "confirm where your tracked position actually is relative "
                              "to your zone polygon coordinates, if zones aren't firing.")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", type=str, default=None)
    args = parser.parse_args()

    if args.target_fps <= 0:
        parser.error("--target-fps must be positive")
    if not (0.0 < args.conf <= 1.0):
        parser.error("--conf must be in (0, 1]")
    if not args.camera_id:
        parser.error("--camera-id cannot be empty")
    if args.dedup_cooldown <= 0:
        parser.error("--dedup-cooldown must be positive")

    try:
        source = resolve_source(args.source)
        classes = parse_classes(args.classes)
    except (FileNotFoundError, ValueError, argparse.ArgumentTypeError) as exc:
        log.error(str(exc))
        return 1

    try:
        zones = load_zones_from_json(args.zones_config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Could not load zones config: %s", exc)
        return 1

    relevant_zones = [z for z in zones if z.camera_id == args.camera_id]
    if not relevant_zones:
        log.warning("No zones configured for camera_id='%s' — the pipeline will run "
                    "but no zone-intrusion alerts will ever fire. Check --camera-id "
                    "matches the camera_id used in your zones config.", args.camera_id)
    elif args.debug_centroids:
        for z in relevant_zones:
            xs = [p[0] for p in z.polygon]
            ys = [p[1] for p in z.polygon]
            log.info("Zone '%s' (%s) bounding box: x=[%.0f, %.0f] y=[%.0f, %.0f]",
                      z.name, z.rule_type, min(xs), max(xs), min(ys), max(ys))

    try:
        tracker = Tracker(model_path=args.model, conf_threshold=args.conf, classes=classes)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        log.error("Could not initialize tracker: %s", exc)
        return 1

    engine = RuleEngine(zones, dedup_cooldown_seconds=args.dedup_cooldown)

    writer = None
    display_active = False
    frame_count = 0
    total_alerts = 0

    log.info("Starting zone demo: source=%s camera_id=%s zones_for_camera=%d",
              mask_credentials(str(args.source)), args.camera_id,
              engine.zone_count_for_camera(args.camera_id))

    try:
        with VideoIngestor(source, target_fps=args.target_fps) as ingestor:
            for frame in ingestor.frames():
                result = tracker.track(frame)
                frame_count += 1

                snapshots = [
                    TrackSnapshot(track_id=t.track_id, centroid=t.centroid,
                                  class_name=t.class_name, confidence=t.confidence)
                    for t in result.tracks
                ]

                if args.debug_centroids and snapshots:
                    positions = ", ".join(
                        f"id={s.track_id} centroid=({s.centroid[0]:.0f},{s.centroid[1]:.0f})"
                        for s in snapshots
                    )
                    log.info("t=%.1fs %s", frame.timestamp, positions)

                events = engine.evaluate(snapshots, camera_id=args.camera_id,
                                          timestamp=frame.timestamp)

                for event in events:
                    total_alerts += 1
                    log.info("🚨 ALERT #%d: %s", total_alerts, event.to_dict())

                annotated = draw_zones(frame.image, zones, args.camera_id)
                annotated = draw_tracks(annotated, result.tracks)
                cv2.putText(annotated, f"alerts: {total_alerts}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

                if args.save_video:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                            args.target_fps, (w, h),
                        )
                        if not writer.isOpened():
                            log.error("Could not open VideoWriter for '%s' — disabling save",
                                      args.save_video)
                            writer = None
                    if writer is not None:
                        writer.write(annotated)

                if args.display:
                    display_active = True
                    cv2.imshow("FrameSentinel — Day 4 (zone-intrusion demo)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("Quit requested by user")
                        break

    except RuntimeError as exc:
        log.error("Ingestion error: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("Interrupted by user — shutting down cleanly")
    finally:
        if writer is not None:
            writer.release()
        if display_active:
            cv2.destroyAllWindows()

    log.info("Done. total_frames=%d total_alerts=%d", frame_count, total_alerts)
    return 0


if __name__ == "__main__":
    sys.exit(main())