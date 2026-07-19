"""
FrameSentinel — Zone Definition + Zone-Intrusion Rule Engine
Day 4

This module is intentionally pure logic: no OpenCV, no Ultralytics, no I/O
in the hot path. It only knows about plain data (points, polygons, track
snapshots) and produces plain AlertEvent objects. That's what makes it
trivially unit-testable in Day 6 with zero mocking, and swappable later
(e.g. if the detector/tracker library changes, this file never has to).

Concepts:
    Zone         — a named polygon on a specific camera, with a rule type.
    TrackSnapshot — the minimal slice of tracker output the rule engine needs
                    (no dependency on track.py's TrackedObject/Ultralytics types).
    AlertEvent   — emitted when a rule fires. This is the same shape that
                    later gets pushed to Redis (Day 7) and written to
                    Postgres (Day 8) — defined once, here, as the contract.
    RuleEngine   — holds zones + per-(track, zone) state, evaluates one
                    frame's tracks against zone-intrusion rules, and returns
                    only the *new* events (edge-triggered on entry — see
                    docs/DECISIONS.md for why this isn't fired every frame).

Usage as a library (see src/run_zone_demo.py for a full wiring example):

    from rule_engine import RuleEngine, Zone, TrackSnapshot, load_zones_from_json

    zones = load_zones_from_json("configs/zones.example.json")
    engine = RuleEngine(zones)

    tracks = [TrackSnapshot(track_id=1, centroid=(120, 340),
                             class_name="person", confidence=0.91)]
    events = engine.evaluate(tracks, camera_id="cam_1", timestamp=12.4)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

log = logging.getLogger("frame-sentinel.rule_engine")

# Rule types supported so far. Loitering/PPE/fall-detection get added to
# this set in later days — keeping it explicit (not a free-text string)
# means a typo in a zone config fails loudly at load time, not silently
# at runtime when the rule just never fires.
SUPPORTED_RULE_TYPES = {"zone_intrusion"}

Point = Tuple[float, float]


# --------------------------------------------------------------------------
# Geometry — pure Python, no numpy/cv2. Ray-casting point-in-polygon.
# --------------------------------------------------------------------------

def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """
    Standard ray-casting algorithm: True if `point` lies inside `polygon`.

    Deliberately implemented without cv2.pointPolygonTest so the rule engine
    has zero dependency on OpenCV — it should be testable and reasoned about
    as plain geometry, not as "a thing that happens to call into a CV library."

    Points ON the boundary may register as inside or outside depending on
    floating point edge behavior — acceptable for this use case (a person
    standing exactly on a drawn line is an edge case no safety system needs
    to be pixel-perfect about).
    """
    if len(polygon) < 3:
        raise ValueError(f"Polygon must have at least 3 points, got {len(polygon)}")

    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        x_intersect = p1x
                    if p1x == p2x or x <= x_intersect:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def validate_polygon(polygon: Sequence[Point], zone_id: str = "") -> None:
    """Raises ValueError with a clear message if the polygon is malformed."""
    if not isinstance(polygon, (list, tuple)):
        raise ValueError(f"Zone '{zone_id}': polygon must be a list of [x, y] points")
    if len(polygon) < 3:
        raise ValueError(
            f"Zone '{zone_id}': polygon needs at least 3 points to enclose an area, "
            f"got {len(polygon)}"
        )
    for i, pt in enumerate(polygon):
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            raise ValueError(f"Zone '{zone_id}': point {i} is not a valid [x, y] pair: {pt!r}")
        x, y = pt
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise ValueError(f"Zone '{zone_id}': point {i} has non-numeric coordinates: {pt!r}")
        if x < 0 or y < 0:
            raise ValueError(
                f"Zone '{zone_id}': point {i} has a negative coordinate {pt!r} — "
                f"pixel coordinates cannot be negative"
            )


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Zone:
    """A named polygon region on a specific camera, with an assigned rule type."""
    zone_id: str
    camera_id: str
    name: str
    polygon: Tuple[Point, ...]  # tuple = immutable, safe to hash/share across frames
    rule_type: str

    def __post_init__(self):
        if not self.zone_id:
            raise ValueError("Zone.zone_id cannot be empty")
        if not self.camera_id:
            raise ValueError("Zone.camera_id cannot be empty")
        if self.rule_type not in SUPPORTED_RULE_TYPES:
            raise ValueError(
                f"Zone '{self.zone_id}': rule_type '{self.rule_type}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_RULE_TYPES)}"
            )
        validate_polygon(self.polygon, self.zone_id)

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)


@dataclass
class TrackSnapshot:
    """
    The minimal slice of tracker output the rule engine needs for one frame.

    Deliberately NOT the same class as track.py's TrackedObject — this keeps
    rule_engine.py free of any import from track.py/ultralytics, so it can be
    unit tested (Day 6) with plain Python objects and no model/GPU involved.
    """
    track_id: int
    centroid: Point
    class_name: str
    confidence: float


@dataclass
class AlertEvent:
    """
    Emitted when a rule transitions from not-triggered to triggered.

    This is the contract other services build on: Day 7 publishes this shape
    to Redis, Day 8 writes it to the `alerts` Postgres table. Defining it once
    here means every downstream consumer agrees on the shape from day one.
    """
    camera_id: str
    zone_id: str
    zone_name: str
    rule: str
    track_id: int
    class_name: str
    confidence: float
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "rule": self.rule,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


# --------------------------------------------------------------------------
# Rule engine
# --------------------------------------------------------------------------

class RuleEngine:
    """
    Evaluates tracked objects against configured zones and emits AlertEvents.

    Zone-intrusion is edge-triggered: an event fires the moment a track
    transitions from outside -> inside a zone, not on every frame it remains
    inside. Firing every frame would mean a person standing in a zone for
    10 seconds at 8fps produces ~80 identical alerts — that's explicitly the
    "alert spam" problem Day 5's dedup logic exists to solve, but there's no
    reason to create the spam in the first place when entry/exit state is
    this cheap to track. Day 5 dedup then handles the harder case: repeated
    entry/exit of the same zone within a short window (e.g. flickering
    detection at a zone boundary).
    """

    def __init__(self, zones: List[Zone]):
        if not isinstance(zones, list):
            raise TypeError("zones must be a list of Zone objects")
        for z in zones:
            if not isinstance(z, Zone):
                raise TypeError(f"Expected Zone instance, got {type(z)}")
        self.zones = zones
        self._zones_by_camera: Dict[str, List[Zone]] = {}
        for z in zones:
            self._zones_by_camera.setdefault(z.camera_id, []).append(z)

        # (track_id, zone_id) -> currently inside (bool)
        self._state: Dict[Tuple[int, str], bool] = {}

    def evaluate(self, tracks: List[TrackSnapshot], camera_id: str,
                 timestamp: float) -> List[AlertEvent]:
        """
        Evaluates one frame's worth of tracks for the given camera.
        Returns only newly-triggered events (empty list is the common case).
        """
        if not camera_id:
            raise ValueError("camera_id cannot be empty")
        if timestamp < 0:
            raise ValueError(f"timestamp cannot be negative, got {timestamp}")

        zones_for_camera = self._zones_by_camera.get(camera_id, [])
        if not zones_for_camera:
            # Not an error — a camera with no configured zones simply never alerts.
            return []

        events: List[AlertEvent] = []
        active_keys: Set[Tuple[int, str]] = set()

        for track in tracks:
            for zone in zones_for_camera:
                if zone.rule_type != "zone_intrusion":
                    continue  # other rule types handled by later days' logic

                key = (track.track_id, zone.zone_id)
                active_keys.add(key)

                is_inside = zone.contains(track.centroid)
                was_inside = self._state.get(key, False)

                if is_inside and not was_inside:
                    events.append(AlertEvent(
                        camera_id=camera_id,
                        zone_id=zone.zone_id,
                        zone_name=zone.name,
                        rule=zone.rule_type,
                        track_id=track.track_id,
                        class_name=track.class_name,
                        confidence=track.confidence,
                        timestamp=timestamp,
                    ))
                    log.info("ALERT: track_id=%s entered zone '%s' (camera=%s) at t=%.1fs",
                             track.track_id, zone.name, camera_id, timestamp)

                self._state[key] = is_inside

        self._prune_stale_state(active_keys)
        return events

    def _prune_stale_state(self, active_keys: Set[Tuple[int, str]]) -> None:
        """
        Drops (track_id, zone_id) state entries for tracks no longer present
        this frame, so a long-running session doesn't accumulate state for
        every track_id that's ever walked through, forever.
        """
        stale = [k for k in self._state if k not in active_keys]
        for k in stale:
            del self._state[k]

    def zone_count_for_camera(self, camera_id: str) -> int:
        return len(self._zones_by_camera.get(camera_id, []))


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------

def load_zones_from_json(path: str) -> List[Zone]:
    """
    Loads zone definitions from a JSON file.

    Expected shape:
    [
      {
        "zone_id": "zone_1",
        "camera_id": "cam_1",
        "name": "Restricted Area A",
        "rule_type": "zone_intrusion",
        "polygon": [[100, 200], [300, 200], [300, 400], [100, 400]]
      }
    ]

    Raises FileNotFoundError / ValueError with a specific, actionable message
    on any malformed input, rather than letting a KeyError/JSONDecodeError
    surface from deep inside this function.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Zone config not found: '{path}'")
    if not config_path.is_file():
        raise ValueError(f"Zone config path is not a file: '{path}'")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read zone config '{path}': {exc}") from exc

    try:
        raw_data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Zone config '{path}' is not valid JSON: {exc}") from exc

    if not isinstance(raw_data, list):
        raise ValueError(f"Zone config '{path}' must be a JSON array of zone objects")

    zones: List[Zone] = []
    seen_ids: Set[str] = set()

    for i, entry in enumerate(raw_data):
        if not isinstance(entry, dict):
            raise ValueError(f"Zone config '{path}': entry {i} is not a JSON object")

        required = {"zone_id", "camera_id", "name", "rule_type", "polygon"}
        missing = required - entry.keys()
        if missing:
            raise ValueError(
                f"Zone config '{path}': entry {i} is missing required field(s): "
                f"{sorted(missing)}"
            )

        zone_id = entry["zone_id"]
        if zone_id in seen_ids:
            raise ValueError(f"Zone config '{path}': duplicate zone_id '{zone_id}'")
        seen_ids.add(zone_id)

        try:
            polygon = tuple(tuple(pt) for pt in entry["polygon"])
            zone = Zone(
                zone_id=zone_id,
                camera_id=entry["camera_id"],
                name=entry["name"],
                polygon=polygon,
                rule_type=entry["rule_type"],
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Zone config '{path}': entry {i} ('{zone_id}') invalid: {exc}") from exc

        zones.append(zone)

    log.info("Loaded %d zone(s) from '%s'", len(zones), path)
    return zones