"""
FrameSentinel — Zone Picker (dev utility)
Day 4

Defining zone polygons by guessing pixel coordinates is slow and error-prone.
This tool grabs the first frame of a video, lets you click points to draw a
polygon, and writes a valid zones.json you can feed straight into
run_zone_demo.py / RuleEngine.

This is a throwaway dev tool, not part of the production pipeline — the
real product replaces this entirely with the drag-to-draw React UI in
Day 12. Kept deliberately simple.

Controls:
    Left click   — add a polygon point
    'z'          — undo last point
    'n'          — finish current polygon, start a new one
    's'          — save all polygons to the output JSON file and quit
    'q' / Esc    — quit without saving

Usage:
    python src/zone_picker.py --source ../sample_videos/sample.mp4 \
        --camera-id cam_1 --output ../configs/zones.example.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2

Point = Tuple[int, int]


class ZonePicker:
    def __init__(self, frame, camera_id: str):
        if frame is None:
            raise ValueError("frame cannot be None — check the video source loaded correctly")
        self.base_frame = frame
        self.camera_id = camera_id
        self.current_polygon: List[Point] = []
        self.completed_polygons: List[List[Point]] = []
        self.window_name = "FrameSentinel — Zone Picker (left-click=add point, n=next, s=save, q=quit)"

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((x, y))

    def _render(self):
        display = self.base_frame.copy()

        for poly in self.completed_polygons:
            self._draw_polygon(display, poly, color=(0, 200, 0), closed=True)

        if self.current_polygon:
            self._draw_polygon(display, self.current_polygon, color=(0, 165, 255), closed=False)

        cv2.putText(display, f"zones drawn: {len(self.completed_polygons)} | "
                              f"current points: {len(self.current_polygon)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return display

    @staticmethod
    def _draw_polygon(image, points: List[Point], color, closed: bool):
        for pt in points:
            cv2.circle(image, pt, 4, color, -1)
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(image, points[i], points[i + 1], color, 2)
            if closed and len(points) > 2:
                cv2.line(image, points[-1], points[0], color, 2)

    def run(self) -> List[List[Point]]:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        try:
            while True:
                cv2.imshow(self.window_name, self._render())
                key = cv2.waitKey(20) & 0xFF

                if key in (ord("q"), 27):  # 27 = Esc
                    print("[zone_picker] quit without saving")
                    return []

                elif key == ord("z"):
                    if self.current_polygon:
                        self.current_polygon.pop()

                elif key == ord("n"):
                    if len(self.current_polygon) >= 3:
                        self.completed_polygons.append(self.current_polygon)
                        self.current_polygon = []
                    else:
                        print("[zone_picker] need at least 3 points before starting a new polygon")

                elif key == ord("s"):
                    if len(self.current_polygon) >= 3:
                        self.completed_polygons.append(self.current_polygon)
                        self.current_polygon = []
                    return self.completed_polygons
        finally:
            cv2.destroyAllWindows()


def polygons_to_zone_configs(polygons: List[List[Point]], camera_id: str,
                              rule_type: str) -> List[dict]:
    zones = []
    for i, polygon in enumerate(polygons, start=1):
        zones.append({
            "zone_id": f"zone_{i}",
            "camera_id": camera_id,
            "name": f"Zone {i}",
            "rule_type": rule_type,
            "polygon": [[x, y] for x, y in polygon],
        })
    return zones


def main() -> int:
    parser = argparse.ArgumentParser(description="FrameSentinel — interactive zone picker")
    parser.add_argument("--source", required=True, help="Video file path")
    parser.add_argument("--camera-id", required=True, help="camera_id to tag zones with")
    parser.add_argument("--rule-type", default="zone_intrusion", choices=["zone_intrusion"])
    parser.add_argument("--output", required=True, help="Path to write zones JSON")
    parser.add_argument("--frame-index", type=int, default=0,
                         help="Which source frame to draw on (default: 0, the first frame). "
                              "Use a frame where your subject is actually visible — "
                              "e.g. --frame-index 40 — otherwise you're drawing a zone "
                              "blind, on an empty room.")
    args = parser.parse_args()

    if args.frame_index < 0:
        print("[zone_picker] ERROR: --frame-index cannot be negative", file=sys.stderr)
        return 1

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[zone_picker] ERROR: source video not found: '{args.source}'", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"[zone_picker] ERROR: could not open video: '{args.source}'", file=sys.stderr)
        return 1

    if args.frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"[zone_picker] ERROR: could not read frame {args.frame_index} from source "
              f"(video may be shorter than that, or index is invalid)", file=sys.stderr)
        return 1

    picker = ZonePicker(frame, camera_id=args.camera_id)
    polygons = picker.run()

    if not polygons:
        print("[zone_picker] no zones drawn — nothing saved")
        return 0

    zone_configs = polygons_to_zone_configs(polygons, args.camera_id, args.rule_type)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(json.dumps(zone_configs, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[zone_picker] ERROR: could not write '{args.output}': {exc}", file=sys.stderr)
        return 1

    print(f"[zone_picker] saved {len(zone_configs)} zone(s) to '{args.output}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())