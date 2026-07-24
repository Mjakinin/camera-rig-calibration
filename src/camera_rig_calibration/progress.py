from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


COUNT_PATTERNS = {
    "registered_images": re.compile(
        r"registered(?:_|\s+)images?\D+(\d+)", re.IGNORECASE
    ),
    "registered_static_cameras": re.compile(
        r"registered(?:_|\s+)static(?:_|\s+)cameras?\D+(\d+)",
        re.IGNORECASE,
    ),
    "selected_moving_frames": re.compile(
        r"selected(?:_|\s+)moving(?:_|\s+)frames?\D+(\d+)",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class ProgressEvent:
    event: str
    job_id: str
    job_index: int
    job_count: int
    stage_id: str
    stage_name: str
    stage_index: int
    stage_count: int
    stage_elapsed_seconds: float
    job_elapsed_seconds: float
    queue_elapsed_seconds: float
    log: str | None = None
    counts: dict[str, int] | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


class ProgressClock:
    def __init__(
        self,
        *,
        job_id: str,
        job_index: int = 1,
        job_count: int = 1,
        queue_started_monotonic: float | None = None,
    ) -> None:
        self.job_id = job_id
        self.job_index = job_index
        self.job_count = job_count
        self.queue_started = queue_started_monotonic or time.monotonic()
        self.job_started = time.monotonic()
        self.stage_started = self.job_started
        self.counts: dict[str, int] = {}

    def begin_stage(self) -> None:
        self.stage_started = time.monotonic()
        self.counts = {}

    def update_counts(self, line: str) -> dict[str, int]:
        for name, pattern in COUNT_PATTERNS.items():
            match = pattern.search(line)
            if match:
                self.counts[name] = int(match.group(1))
        return dict(self.counts)

    def event(
        self,
        *,
        event: str,
        stage_id: str,
        stage_name: str,
        stage_index: int,
        stage_count: int,
        log: Path | None = None,
    ) -> ProgressEvent:
        now = time.monotonic()
        return ProgressEvent(
            event=event,
            job_id=self.job_id,
            job_index=self.job_index,
            job_count=self.job_count,
            stage_id=stage_id,
            stage_name=stage_name,
            stage_index=stage_index,
            stage_count=stage_count,
            stage_elapsed_seconds=now - self.stage_started,
            job_elapsed_seconds=now - self.job_started,
            queue_elapsed_seconds=now - self.queue_started,
            log=str(log) if log is not None else None,
            counts=dict(self.counts),
        )


def terminal_lines(event: ProgressEvent) -> list[str]:
    lines = [
        (
            f"[{event.job_id}] Job {event.job_index}/{event.job_count}, "
            f"Step {event.stage_index}/{event.stage_count}: {event.stage_name}"
        ),
        f"Stage elapsed: {event.stage_elapsed_seconds:.1f} s",
        f"Job elapsed: {event.job_elapsed_seconds:.1f} s",
        f"Queue elapsed: {event.queue_elapsed_seconds:.1f} s",
    ]
    for key, value in sorted((event.counts or {}).items()):
        lines.append(f"{key.replace('_', ' ').title()}: {value}")
    if event.log:
        lines.append(f"Log: {event.log}")
    return lines
