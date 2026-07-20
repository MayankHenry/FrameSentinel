"""
FrameSentinel — Zone Rules + Loitering + Alert Deduplication
Day 4 (zone-intrusion) + Day 5 (loitering, dedup)

This module is intentionally pure logic: no OpenCV, no Ultralytics, no I/O
in the hot path. It only knows about plain data (points, polygons, track
snapshots) and produces plain AlertEvent objects. That's what makes it
trivially unit-testable in Day 6 with zero mocking, and swappable later
(e.g. if the detector/tracker library changes, this file never has to).

Concepts:
    Zone            — a named polygon on a specific camera, with a rule type
                       and (for loitering) a dwell threshold + movement tolerance.
    TrackSnapshot    — the minimal slice of tracker output the rule engine needs.
    AlertEvent       — emitted when a rule fires. Same shape gets pushed to
                       Redis (Day 7) and written to Postgres (Day 8).
    AlertDeduplicator — cooldown-window gate in front of every emitted event.
                       In-memory today; the interface is designed so a
                       Redis-backed implementation can drop in later without
                       RuleEngine changing at all (per the project plan).
    RuleEngine       — holds zones + per-(track, zone) state, evaluates one
                       frame's tracks, and returns only newly-fired events.

Rule types:
    zone_intrusion — edge-triggered: fires once on outside->inside transition.
    loitering      — fires once per continuous "stationary dwell" that exceeds
                     zone.loiter_seconds, where "stationary" means the track's
                     centroid hasn't moved more than zone.movement_threshold_px
                     since the dwell began. Leaving the zone, or moving beyond
                     the tolerance, resets the dwell timer.

Every emitted event — regardless of rule type — passes through the
AlertDeduplicator before being returned, as a second line of defense against
alert spam (e.g. a flickering detection at a zone boundary re-triggering
zone_intrusion many times in a few seconds).

Usage as a library (see src/run_zone_demo.py for a full wiring example):

    from rule_engine import RuleEngine, Zone, TrackSnapshot, load_zones_from_json

    zones = load_zones_from_json("configs/zones.example.json")
    engine = RuleEngine(zones, dedup_cooldown_seconds=30.0)

    tracks = [TrackSnapshot(track_id=1, centroid=(120, 340),
                             class_name="person", confidence=0.91)]
    events = engine.evaluate(tracks, camera_id="cam_1", timestamp=12.4)
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

log = logging.getLogger("frame-sentinel.rule_engine")

# Rule types supported so far. PPE/fall-detection get added to this set in
# later days — keeping it explicit (not a free-text string) means a typo in
# a zone config fails loudly at load time, not silently at runtime when the
# rule just never fires.
SUPPORTED_RULE_TYPES = {"zone_intrusion", "loitering"}

# Sane defaults for loitering, used when a zone config omits the optional field.
DEFAULT_MOVEMENT_THRESHOLD_PX = 30.0
DEFAULT_DEDUP_COOLDOWN_SECONDS = 30.0

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


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _orientation(a: Point, b: Point, c: Point) -> int:
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    """True if c lies on segment a-b, given a/b/c are already known collinear."""
    return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Standard orientation-based segment intersection test."""
    o1, o2 = _orientation(p1, p2, p3), _orientation(p1, p2, p4)
    o3, o4 = _orientation(p3, p4, p1), _orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, p3):
        return True
    if o2 == 0 and _on_segment(p1, p2, p4):
        return True
    if o3 == 0 and _on_segment(p3, p4, p1):
        return True
    if o4 == 0 and _on_segment(p3, p4, p2):
        return True
    return False


def is_simple_polygon(polygon: Sequence[Point]) -> bool:
    """
    Checks that no two non-adjacent edges of the polygon cross each other.

    A self-intersecting polygon almost always means the points don't describe
    one coherent region — the most common real-world cause (hit during this
    project's own testing) is clicking points for what was meant to be
    several separate zones without starting a new polygon in between, so
    unrelated clusters of points got joined into one shape by the closing
    edges. point_in_polygon's ray-casting still returns *an* answer for a
    self-intersecting shape, just not a meaningful or predictable one.
    """
    n = len(polygon)
    edges = [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue  # adjacent edges legitimately share a vertex
            a1, a2 = edges[i]
            b1, b2 = edges[j]
            if _segments_intersect(a1, a2, b1, b2):
                return False
    return True


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

    if len(polygon) >= 4 and not is_simple_polygon(polygon):
        raise ValueError(
            f"Zone '{zone_id}': polygon is self-intersecting (its edges cross "
            f"themselves) — this usually means multiple separate zones got "
            f"merged into one polygon. If you used zone_picker.py, make sure "
            f"you press 'n' to start a NEW polygon before clicking points for "
            f"a different zone — otherwise all clicks in one session become a "
            f"single (often self-intersecting) shape."
        )


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Zone:
    """
    A named polygon region on a specific camera, with an assigned rule type.

    loiter_seconds / movement_threshold_px only apply to rule_type="loitering"
    and are validated accordingly — required for loitering zones, ignored
    (and must be omitted or None) for zone_intrusion zones, to avoid a config
    silently having no effect because it was set on the wrong rule type.
    """
    zone_id: str
    camera_id: str
    name: str
    polygon: Tuple[Point, ...]  # tuple = immutable, safe to hash/share across frames
    rule_type: str
    loiter_seconds: Optional[float] = None
    movement_threshold_px: float = DEFAULT_MOVEMENT_THRESHOLD_PX

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

        if self.rule_type == "loitering":
            if self.loiter_seconds is None:
                raise ValueError(
                    f"Zone '{self.zone_id}': rule_type='loitering' requires "
                    f"'loiter_seconds' to be set"
                )
            if self.loiter_seconds <= 0:
                raise ValueError(
                    f"Zone '{self.zone_id}': loiter_seconds must be positive, "
                    f"got {self.loiter_seconds}"
                )
            if self.movement_threshold_px <= 0:
                raise ValueError(
                    f"Zone '{self.zone_id}': movement_threshold_px must be positive, "
                    f"got {self.movement_threshold_px}"
                )
        elif self.rule_type == "zone_intrusion" and self.loiter_seconds is not None:
            raise ValueError(
                f"Zone '{self.zone_id}': loiter_seconds is set but rule_type is "
                f"'zone_intrusion' — this field only applies to 'loitering' zones "
                f"and would silently have no effect here"
            )

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
    Emitted when a rule fires (after passing the deduplicator).

    This is the contract other services build on: Day 7 publishes this shape
    to Redis, Day 8 writes it to the `alerts` Postgres table. Defined once
    here so every downstream consumer agrees on the shape from day one.
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


@dataclass
class _LoiterState:
    """Internal per-(track, zone) bookkeeping for the loitering rule."""
    reference_centroid: Point
    stationary_since: float
    alerted: bool = False


# --------------------------------------------------------------------------
# Alert deduplication
# --------------------------------------------------------------------------

class AlertDeduplicator(ABC):
    """
    Cooldown-window gate for emitted alerts, keyed by (track_id, zone_id, rule).

    Abstract so a Redis-backed implementation (sharing dedup state across
    multiple inference-service processes/cameras) can be substituted later
    without RuleEngine changing at all — it only ever calls should_emit().
    """

    @abstractmethod
    def should_emit(self, key: str, timestamp: float) -> bool:
        """Returns True if an alert for this key may be emitted now, and
        records that emission. Returns False if still within cooldown."""
        raise NotImplementedError


class InMemoryDeduplicator(AlertDeduplicator):
    """
    Default dedup implementation — a plain dict of last-emitted timestamps.
    Fine for a single-process resume project; swap for RedisDeduplicator
    (same interface) when running multiple inference workers that need to
    share dedup state, per docs/DECISIONS.md.
    """

    def __init__(self, cooldown_seconds: float = DEFAULT_DEDUP_COOLDOWN_SECONDS):
        if cooldown_seconds <= 0:
            raise ValueError(f"cooldown_seconds must be positive, got {cooldown_seconds}")
        self.cooldown_seconds = cooldown_seconds
        self._last_emitted: Dict[str, float] = {}

    def should_emit(self, key: str, timestamp: float) -> bool:
        if not key:
            raise ValueError("dedup key cannot be empty")
        last = self._last_emitted.get(key)
        if last is None or (timestamp - last) >= self.cooldown_seconds:
            self._last_emitted[key] = timestamp
            return True
        return False

    def prune_older_than(self, timestamp: float, max_age_seconds: float) -> None:
        """Keeps the dict from growing unbounded over a long-running session."""
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        stale = [k for k, t in self._last_emitted.items() if timestamp - t > max_age_seconds]
        for k in stale:
            del self._last_emitted[k]

    def size(self) -> int:
        return len(self._last_emitted)


# --------------------------------------------------------------------------
# Rule engine
# --------------------------------------------------------------------------

class RuleEngine:
    """
    Evaluates tracked objects against configured zones and emits AlertEvents.

    zone_intrusion is edge-triggered: fires once on the outside->inside
    transition, not every frame the track remains inside.

    loitering fires once per continuous stationary dwell that exceeds
    zone.loiter_seconds. "Stationary" allows movement up to
    zone.movement_threshold_px from a reference point before the dwell timer
    resets — real people sway, shift weight, and jitter slightly even when
    "standing still," so a zero-tolerance check would almost never fire.

    Every candidate event (from either rule) passes through the
    AlertDeduplicator as a final safety net before being returned.
    """

    def __init__(self, zones: List[Zone], deduplicator: Optional[AlertDeduplicator] = None,
                 dedup_cooldown_seconds: float = DEFAULT_DEDUP_COOLDOWN_SECONDS):
        if not isinstance(zones, list):
            raise TypeError("zones must be a list of Zone objects")
        for z in zones:
            if not isinstance(z, Zone):
                raise TypeError(f"Expected Zone instance, got {type(z)}")

        self.zones = zones
        self._zones_by_camera: Dict[str, List[Zone]] = {}
        for z in zones:
            self._zones_by_camera.setdefault(z.camera_id, []).append(z)

        # (track_id, zone_id) -> currently inside the zone (bool)
        self._state: Dict[Tuple[int, str], bool] = {}
        # (track_id, zone_id) -> loitering dwell bookkeeping
        self._loiter_state: Dict[Tuple[int, str], _LoiterState] = {}

        self.deduplicator = deduplicator or InMemoryDeduplicator(dedup_cooldown_seconds)

        self._evaluate_call_count = 0

    def evaluate(self, tracks: List[TrackSnapshot], camera_id: str,
                 timestamp: float) -> List[AlertEvent]:
        """
        Evaluates one frame's worth of tracks for the given camera.
        Returns only newly-triggered, non-deduplicated events (empty list is
        the common case).
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
                key = (track.track_id, zone.zone_id)
                active_keys.add(key)

                is_inside = zone.contains(track.centroid)
                was_inside = self._state.get(key, False)

                if zone.rule_type == "zone_intrusion":
                    if is_inside and not was_inside:
                        self._try_emit(events, zone, track, timestamp)

                elif zone.rule_type == "loitering":
                    if is_inside:
                        self._evaluate_loitering(events, zone, track, timestamp, key)
                    else:
                        # Leaving the zone resets the dwell entirely — re-entry
                        # starts a fresh loitering window, it doesn't resume one.
                        self._loiter_state.pop(key, None)

                self._state[key] = is_inside

        self._prune_stale_state(active_keys)

        # Periodic dedup cleanup so long sessions don't grow the dict forever.
        # Every 200 evaluate() calls is arbitrary but cheap and frequent enough.
        self._evaluate_call_count += 1
        if isinstance(self.deduplicator, InMemoryDeduplicator) and self._evaluate_call_count % 200 == 0:
            self.deduplicator.prune_older_than(timestamp, max_age_seconds=self.deduplicator.cooldown_seconds * 10)

        return events

    def _evaluate_loitering(self, events: List[AlertEvent], zone: Zone,
                             track: TrackSnapshot, timestamp: float,
                             key: Tuple[int, str]) -> None:
        state = self._loiter_state.get(key)

        if state is None:
            # First frame seen inside this zone for this track — start the clock.
            self._loiter_state[key] = _LoiterState(
                reference_centroid=track.centroid, stationary_since=timestamp
            )
            return

        distance = euclidean_distance(state.reference_centroid, track.centroid)
        if distance > zone.movement_threshold_px:
            # Moved enough that this isn't the same "standing still" episode —
            # restart the dwell timer from here.
            state.reference_centroid = track.centroid
            state.stationary_since = timestamp
            state.alerted = False
            return

        dwell_duration = timestamp - state.stationary_since
        if dwell_duration >= zone.loiter_seconds and not state.alerted:
            self._try_emit(events, zone, track, timestamp)
            state.alerted = True  # don't re-check this dwell again once fired

    def _try_emit(self, events: List[AlertEvent], zone: Zone, track: TrackSnapshot,
                  timestamp: float) -> None:
        """Builds the candidate event and gates it through the deduplicator."""
        dedup_key = f"{track.track_id}:{zone.zone_id}:{zone.rule_type}"
        if not self.deduplicator.should_emit(dedup_key, timestamp):
            log.debug("Suppressed duplicate alert (dedup cooldown active): %s", dedup_key)
            return

        event = AlertEvent(
            camera_id=zone.camera_id,
            zone_id=zone.zone_id,
            zone_name=zone.name,
            rule=zone.rule_type,
            track_id=track.track_id,
            class_name=track.class_name,
            confidence=track.confidence,
            timestamp=timestamp,
        )
        events.append(event)
        log.info("ALERT: track_id=%s rule=%s zone='%s' (camera=%s) at t=%.1fs",
                  track.track_id, zone.rule_type, zone.name, zone.camera_id, timestamp)

    def _prune_stale_state(self, active_keys: Set[Tuple[int, str]]) -> None:
        """
        Drops (track_id, zone_id) state entries for tracks no longer present
        this frame, so a long-running session doesn't accumulate state for
        every track_id that's ever walked through, forever.
        """
        stale = [k for k in self._state if k not in active_keys]
        for k in stale:
            del self._state[k]
            self._loiter_state.pop(k, None)

    def zone_count_for_camera(self, camera_id: str) -> int:
        return len(self._zones_by_camera.get(camera_id, []))


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------

def load_zones_from_json(path: str) -> List[Zone]:
    """
    Loads zone definitions from a JSON file.

    Expected shape (loitering fields only required when rule_type="loitering"):
    [
      {
        "zone_id": "zone_1",
        "camera_id": "cam_1",
        "name": "Restricted Area A",
        "rule_type": "zone_intrusion",
        "polygon": [[100, 200], [300, 200], [300, 400], [100, 400]]
      },
      {
        "zone_id": "zone_2",
        "camera_id": "cam_1",
        "name": "Loading Dock",
        "rule_type": "loitering",
        "polygon": [[400, 100], [600, 100], [600, 300], [400, 300]],
        "loiter_seconds": 10,
        "movement_threshold_px": 30
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
        # utf-8-sig transparently strips a UTF-8 BOM if present (e.g. from
        # PowerShell's `Out-File -Encoding utf8`) and behaves identically to
        # utf-8 for files without one — safe default for a config file that
        # may be hand-edited on Windows.
        raw_text = config_path.read_text(encoding="utf-8-sig")
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

        rule_type = entry["rule_type"]
        if rule_type == "loitering" and "loiter_seconds" not in entry:
            raise ValueError(
                f"Zone config '{path}': entry {i} ('{zone_id}') has "
                f"rule_type='loitering' but is missing 'loiter_seconds'"
            )

        try:
            polygon = tuple(tuple(pt) for pt in entry["polygon"])
            zone = Zone(
                zone_id=zone_id,
                camera_id=entry["camera_id"],
                name=entry["name"],
                polygon=polygon,
                rule_type=rule_type,
                loiter_seconds=entry.get("loiter_seconds"),
                movement_threshold_px=entry.get("movement_threshold_px",
                                                 DEFAULT_MOVEMENT_THRESHOLD_PX),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Zone config '{path}': entry {i} ('{zone_id}') invalid: {exc}") from exc

        zones.append(zone)

    log.info("Loaded %d zone(s) from '%s'", len(zones), path)
    return zones