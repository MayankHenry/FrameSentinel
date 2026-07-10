"""
FrameSentinel — Video Ingestion + Frame Sampling
Day 1

Reads a video source (file path, webcam index, or RTSP URL later) and yields
sampled frames at a configurable target FPS, regardless of the source's
native FPS. This decoupling matters once real RTSP cameras are in the mix —
you sample down for inference load, not because the camera is slow.

Usage:
    python src/ingest.py --source sample_videos/warehouse_demo.mp4 --target-fps 8
    python src/ingest.py --source 0 --target-fps 8       # webcam
"""

import argparse
import time
from dataclasses import dataclass
from typing import Iterator, Union

import cv2


@dataclass
class Frame:
    """A single sampled frame plus its source metadata."""
    index: int          # sampled-frame index (not raw source frame index)
    source_index: int   # original frame index in the source stream
    timestamp: float     # seconds since ingestion started
    image: "cv2.Mat"     # BGR numpy array


class VideoIngestor:
    """
    Wraps cv2.VideoCapture and yields frames sampled at target_fps.

    Works today with a local file or webcam index. RTSP is just a different
    `source` string (e.g. "rtsp://user:pass@host:554/stream") — cv2.VideoCapture
    handles the URI the same way, so this class does not need to change when
    real cameras are introduced. That's the point of building it this way now.
    """

    def __init__(self, source: Union[str, int], target_fps: float = 8.0):
        self.source = source
        self.target_fps = target_fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoIngestor":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def frames(self) -> Iterator[Frame]:
        """
        Yields Frame objects sampled at self.target_fps.

        Sampling strategy: compute how many source frames to skip based on
        the source's native FPS vs target_fps, then grab-and-discard those
        frames rather than decoding every one. cv2's .grab() is cheap
        compared to .retrieve()/.read(), so this keeps CPU load down on
        high-FPS sources.
        """
        if self._cap is None:
            raise RuntimeError("Call open() or use as a context manager first.")

        source_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Guard against sources that report 0 FPS (some webcams/streams do)
        source_fps = source_fps if source_fps > 0 else 30.0

        frame_interval = max(1, round(source_fps / self.target_fps))

        start_time = time.time()
        source_index = 0
        sampled_index = 0

        while True:
            ok = self._cap.grab()
            if not ok:
                break  # end of file or stream drop

            if source_index % frame_interval == 0:
                ok, image = self._cap.retrieve()
                if not ok:
                    break
                yield Frame(
                    index=sampled_index,
                    source_index=source_index,
                    timestamp=time.time() - start_time,
                    image=image,
                )
                sampled_index += 1

            source_index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="FrameSentinel — Day 1 ingestion smoke test")
    parser.add_argument("--source", required=True,
                         help="Video file path, webcam index (e.g. 0), or RTSP URL")
    parser.add_argument("--target-fps", type=float, default=8.0,
                         help="Frames per second to sample (default: 8)")
    parser.add_argument("--display", action="store_true",
                         help="Show sampled frames in a window (press 'q' to quit)")
    args = parser.parse_args()

    # allow webcam index passed as a string int, e.g. "--source 0"
    source: Union[str, int] = int(args.source) if args.source.isdigit() else args.source

    with VideoIngestor(source, target_fps=args.target_fps) as ingestor:
        count = 0
        last_report = time.time()
        for frame in ingestor.frames():
            count += 1

            if args.display:
                cv2.imshow("FrameSentinel — Day 1 (raw sampled frames)", frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Lightweight throughput log every ~1s, useful once real cameras
            # are in play and you're checking for dropped frames.
            if time.time() - last_report >= 1.0:
                print(f"[ingest] sampled_frames={count} "
                      f"last_source_index={frame.source_index} "
                      f"t={frame.timestamp:.1f}s")
                last_report = time.time()

        print(f"[ingest] done. total sampled frames: {count}")

    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
