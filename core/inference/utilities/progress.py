from operator import length_hint
from typing import Iterable

from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Column
from rich.text import Text


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

    from core.shared_dependent import progress

    def track() -> Iterable:
        total = length_hint(iterable)
        task_id = progress.add_task("Working...", total=total)
        for v in iterable:
            yield v
            progress.advance(task_id, 1)
            progress.refresh()
        progress.remove_task(task_id)

    return track()


class MultiProgress(Progress):
    "Progress bar that can show multiple tasks at once"

    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("progress_type") == "download":
                self.columns = (
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

            else:
                self.columns = (
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
            yield self.make_tasks_table([task])

    def advance(self, task_id: TaskID, advance: float = 1) -> None:
        super().advance(task_id, advance)
        super().refresh()
