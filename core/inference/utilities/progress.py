from typing import Iterable
from operator import length_hint

from rich.table import Column
from rich.text import Text
from rich.progress import (
    BarColumn,
    DownloadColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    ProgressColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


class _TimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def __init__(
        self, compact: bool = False, table_column: Column | None = None
    ) -> None:
        self.compact = compact
        super().__init__(table_column)

    def render(self, task):
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text(
                "--:--" if self.compact else "-:--:--", style="progress.elapsed"
            )
        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)
        if self.compact and not hours:
            return Text(f"{minutes:02d}:{seconds:02d}", style="progress.elapsed")
        return Text(f"{hours:d}:{minutes:02d}:{seconds:02d}", style="progress.elapsed")


class _SpeedColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task):
        """Show time elapsed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        unit = "it/s" if speed >= 1 else "s/it"
        return Text(f"{speed:.02f}{unit}", style="progress.percentage")


def progress_bar(iterable: Iterable) -> Iterable:
    "Create and start the Progress thread and then stop after iterating"

    p = new_progress()
    p.start()

    def track() -> Iterable:
        total = length_hint(iterable)
        task_id = p.add_task("Working...", total=total)
        for v in iterable:
            yield v
            p.advance(task_id, 1)
            p.refresh()
        p.stop()

    return track()


def new_progress(file: bool = False) -> Progress:
    "Create a new Progress object"

    if file:
        return Progress(
            TaskProgressColumn(),
            BarColumn(None),
            DownloadColumn(),
            TextColumn("•"),
            _TimeElapsedColumn(compact=True),
            TextColumn("|"),
            TimeRemainingColumn(compact=True),
            TextColumn("•"),
            TransferSpeedColumn(),
        )
    return Progress(
        TaskProgressColumn(),
        BarColumn(None),
        MofNCompleteColumn(),
        TextColumn("•"),
        _TimeElapsedColumn(compact=True),
        TextColumn("|"),
        TimeRemainingColumn(compact=True),
        TextColumn("•"),
        _SpeedColumn(),
    )
