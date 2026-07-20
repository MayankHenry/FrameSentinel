"""
FrameSentinel — Zone Picker (dev utility)
Day 4 (zone_intrusion) + Day 5 (loitering support, --append mode)

Defining zone polygons by guessing pixel coordinates is slow and error-prone.
This tool grabs a frame of a video, lets you click points to draw a polygon,
and writes a valid zones.json you can feed straight into run_zone_demo.py /
RuleEngine.

This is a throwaway dev tool, not part of the production pipeline — the
real product replaces this entirely with the drag-to-draw React UI in
Day 12. Kept deliberately simple.

Controls:
    Left click   — add a polygon point (click each CORNER once — do NOT
                   click back on your starting point to "close" the shape;
                   the polygon closes automatically between your last click
                   and your first one)
    'z'          — undo last point
    'n'          — finish current polygon, start a new one (REQUIRED between
                   distinct zones — if you draw one shape, then click points
                   for a second shape without pressing 'n' first, both sets
                   of points become ONE polygon, which is almost always
                   self-intersecting and meaningless)
    's'          — save all polygons to the output JSON file and quit
    'q' / Esc    — quit without saving

Usage:
    # zone_intrusion (default) — same as Day 4
    python src/zone_picker.py --source ../sample_videos/sample.mp4 \
        --camera-id cam_1 --output ../configs/zones.example.json --frame-index 40

    # loitering zone, ADDED to an existing config instead of overwriting it
    python src/zone_picker.py --source ../sample_videos/sample.mp4 \
        --camera-id cam_1 --output ../configs/zones.example.json --frame-index 40 \
        --rule-type loitering --loiter-seconds 10 --append

Without --append, running this tool OVERWRITES --output entirely — it has no
memory of zones written by a previous run. This bit us once already (see
docs/DECISIONS.md): draw one zone type, quit, then re-run to draw the other
and the first one vanishes unless --append is used.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

Point = Tuple[int, int]


CLOSE_CLICK_GUARD_PX = 15  # ignore clicks this close to the polygon's start point


class ZonePicker:
    def __init__(self, frame, camera_id: str):
        if frame is None:
            raise ValueError("frame cannot be None — check the video source loaded correctly")
        self.base_frame = frame
        self.camera_id = camera_id
        self.current_polygon: List[Point] = []
        self.completed_polygons: List[List[Point]] = []
        self.status_message = ""
        self.window_name = "FrameSentinel — Zone Picker (left-click=add point, n=next, s=save, q=quit)"

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.current_polygon) >= 3:
            start_x, start_y = self.current_polygon[0]
            if ((x - start_x) ** 2 + (y - start_y) ** 2) ** 0.5 < CLOSE_CLICK_GUARD_PX:
                # This is almost certainly an attempt to "close the loop" by
                # re-clicking the start point — the polygon already closes
                # automatically, and a near-duplicate point here is exactly
                # what caused a real self-intersection bug during testing.
                self.status_message = ("Ignored click near start point — the shape "
                                        "closes automatically. Press 's' or 'n' instead.")
                return

        self.current_polygon.append((x, y))
        self.status_message = ""

    def _render(self):
        display = self.base_frame.copy()

        for poly in self.completed_polygons:
            self._draw_polygon(display, poly, color=(0, 200, 0), closed=True)

        if self.current_polygon:
            self._draw_polygon(display, self.current_polygon, color=(0, 165, 255), closed=False)
            # Visual cue for the no-reclick guard radius around the start point
            cv2.circle(display, self.current_polygon[0], CLOSE_CLICK_GUARD_PX, (0, 0, 255), 1)

        cv2.putText(display, f"zones drawn: {len(self.completed_polygons)} | "
                              f"current points: {len(self.current_polygon)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.status_message:
            cv2.putText(display, self.status_message, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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

    def _try_finalize_current(self) -> bool:
        """
        Validates the in-progress polygon before moving it to completed_polygons.
        Returns False (and sets a status message) if it's malformed, leaving
        current_polygon untouched so the user can 'z' (undo) the bad point(s).
        """
        if len(self.current_polygon) < 3:
            self.status_message = "Need at least 3 points to finish a polygon"
            return False

        # Imported here (not at module top) to keep zone_picker.py's only
        # hard dependency cv2 for the interactive parts — rule_engine's
        # validation is reused, not duplicated, since it's the same check
        # run_zone_demo.py will apply anyway. Failing here, before saving,
        # is strictly better than failing later at load time.
        from rule_engine import is_simple_polygon

        if not is_simple_polygon(self.current_polygon):
            self.status_message = ("This shape self-intersects (crosses itself) — "
                                    "press 'z' to undo points until it's a clean "
                                    "shape, or check you didn't merge two zones "
                                    "into one without pressing 'n'.")
            return False

        self.completed_polygons.append(self.current_polygon)
        self.current_polygon = []
        self.status_message = ""
        return True

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
                        self.status_message = ""

                elif key == ord("n"):
                    if self._try_finalize_current():
                        print(f"[zone_picker] zone {len(self.completed_polygons)} "
                              f"finished ({len(self.completed_polygons[-1])} points) — "
                              f"draw the next one")
                    else:
                        print(f"[zone_picker] {self.status_message}")

                elif key == ord("s"):
                    if self.current_polygon:
                        if not self._try_finalize_current():
                            print(f"[zone_picker] {self.status_message}")
                            continue
                    if not self.completed_polygons:
                        print("[zone_picker] no completed zones to save yet")
                        continue
                    return self.completed_polygons
        finally:
            cv2.destroyAllWindows()


def next_zone_id(existing: List[dict]) -> int:
    """Finds the next free numeric suffix for 'zone_N' given existing zone_ids,
    so --append never collides with zones from a previous run."""
    max_n = 0
    pattern = re.compile(r"^zone_(\d+)$")
    for entry in existing:
        m = pattern.match(str(entry.get("zone_id", "")))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def polygons_to_zone_configs(polygons: List[List[Point]], camera_id: str,
                              rule_type: str, start_index: int,
                              loiter_seconds: Optional[float],
                              movement_threshold_px: float) -> List[dict]:
    zones = []
    for offset, polygon in enumerate(polygons):
        i = start_index + offset
        zone = {
            "zone_id": f"zone_{i}",
            "camera_id": camera_id,
            "name": f"Zone {i}",
            "rule_type": rule_type,
            "polygon": [[x, y] for x, y in polygon],
        }
        if rule_type == "loitering":
            zone["loiter_seconds"] = loiter_seconds
            zone["movement_threshold_px"] = movement_threshold_px
        zones.append(zone)
    return zones


def main() -> int:
    parser = argparse.ArgumentParser(description="FrameSentinel — interactive zone picker")
    parser.add_argument("--source", required=True, help="Video file path")
    parser.add_argument("--camera-id", required=True, help="camera_id to tag zones with")
    parser.add_argument("--rule-type", default="zone_intrusion",
                         choices=["zone_intrusion", "loitering"])
    parser.add_argument("--loiter-seconds", type=float, default=None,
                         help="Required when --rule-type loitering")
    parser.add_argument("--movement-threshold-px", type=float, default=30.0,
                         help="Max centroid drift (px) still considered 'stationary' "
                              "for loitering zones (default: 30)")
    parser.add_argument("--output", required=True, help="Path to write zones JSON")
    parser.add_argument("--append", action="store_true",
                         help="Add to the zones already in --output instead of "
                              "overwriting the file. Without this flag, the output "
                              "file is REPLACED entirely.")
    parser.add_argument("--frame-index", type=int, default=0,
                         help="Which source frame to draw on (default: 0, the first frame). "
                              "Use a frame where your subject is actually visible — "
                              "e.g. --frame-index 40 — otherwise you're drawing a zone "
                              "blind, on an empty room.")
    args = parser.parse_args()

    if args.frame_index < 0:
        print("[zone_picker] ERROR: --frame-index cannot be negative", file=sys.stderr)
        return 1

    if args.rule_type == "loitering" and args.loiter_seconds is None:
        print("[zone_picker] ERROR: --rule-type loitering requires --loiter-seconds",
              file=sys.stderr)
        return 1
    if args.loiter_seconds is not None and args.loiter_seconds <= 0:
        print("[zone_picker] ERROR: --loiter-seconds must be positive", file=sys.stderr)
        return 1
    if args.movement_threshold_px <= 0:
        print("[zone_picker] ERROR: --movement-threshold-px must be positive", file=sys.stderr)
        return 1

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[zone_picker] ERROR: source video not found: '{args.source}'", file=sys.stderr)
        return 1

    # Load existing zones up front if appending, so we fail fast on a corrupt
    # existing file rather than after the person has just spent time drawing.
    existing_zones: List[dict] = []
    output_path = Path(args.output)
    if args.append and output_path.exists():
        try:
            existing_zones = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(existing_zones, list):
                raise ValueError("existing zones file must contain a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[zone_picker] ERROR: --append given but '{args.output}' is not a "
                  f"valid zones JSON file: {exc}", file=sys.stderr)
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

    start_index = next_zone_id(existing_zones) if args.append else 1
    new_zones = polygons_to_zone_configs(
        polygons, args.camera_id, args.rule_type, start_index,
        args.loiter_seconds, args.movement_threshold_px,
    )

    all_zones = existing_zones + new_zones if args.append else new_zones

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(json.dumps(all_zones, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[zone_picker] ERROR: could not write '{args.output}': {exc}", file=sys.stderr)
        return 1

    if args.append:
        print(f"[zone_picker] appended {len(new_zones)} zone(s) — "
              f"'{args.output}' now has {len(all_zones)} total")
    else:
        print(f"[zone_picker] saved {len(new_zones)} zone(s) to '{args.output}' "
              f"(any previous contents were replaced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())