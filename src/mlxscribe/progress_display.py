"""Terminal progress bars: one bar for a whole job, however it is split internally."""

from __future__ import annotations

import contextlib
import importlib
import sys
from typing import Iterator

import tqdm

# mlx-whisper counts progress in mel frames: 100 per second of audio.
WHISPER_FRAMES_PER_SECOND = 100


def format_clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class _ClockBar(tqdm.tqdm):
    """A tqdm bar that can show its position as audio time, via ``{clock}``."""

    @property
    def format_dict(self) -> dict:
        values = super().format_dict
        values["clock"] = f"{format_clock(values['n'])}/{format_clock(values['total'] or 0)}"
        return values


class JobProgress:
    """A single bar over a job's total work, drawn only when stderr is a terminal.

    Messages printed with ``note`` appear above the bar instead of breaking it.
    """

    def __init__(self, total: float, *, desc: str, unit: str, clock: bool = False):
        self.clock = clock
        self.total = total
        self.position = 0.0  # tracked here: a disabled tqdm bar does not count
        self.bar = _ClockBar(
            total=total,
            desc=desc,
            unit=unit,
            disable=None,  # off when stderr is not a TTY, so logs stay clean
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {clock} [{elapsed}<{remaining}]"
                if clock
                else "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]"
            ),
        )

    def advance(self, amount: float) -> None:
        step = max(0.0, min(amount, self.total - self.position))
        self.position += step
        self.bar.update(step)

    def advance_to(self, position: float) -> None:
        self.advance(position - self.position)

    def note(self, message: str) -> None:
        if self.bar.disable:
            print(message)
        else:
            tqdm.tqdm.write(message, file=sys.stderr)

    def close(self) -> None:
        self.bar.close()


class _ForwardingBar:
    """Stands in for mlx-whisper's own tqdm bar and forwards its updates to a JobProgress."""

    def __init__(self, target: JobProgress | None, *args, **kwargs):
        self.target = target

    def __enter__(self) -> "_ForwardingBar":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def update(self, frames: float = 1) -> None:
        if self.target is not None:
            self.target.advance(frames / WHISPER_FRAMES_PER_SECOND)

    def close(self) -> None:
        return None


@contextlib.contextmanager
def whisper_progress(target: JobProgress | None) -> Iterator[None]:
    """Route mlx-whisper's per-call progress bar into ``target``, or silence it with None.

    mlx-whisper draws a fresh bar for every call, so a chunked job would show one bar per
    chunk. This swaps the ``tqdm`` it uses for the length of the block. If a future
    mlx-whisper stops using ``tqdm.tqdm`` there, this quietly does nothing.
    """
    try:
        # Not `import mlx_whisper.transcribe`: the package re-exports a function with the
        # module's name, which shadows the module on attribute access.
        whisper_module = importlib.import_module("mlx_whisper.transcribe")
    except ImportError:
        yield
        return
    original = getattr(whisper_module, "tqdm", None)
    if original is None or not hasattr(original, "tqdm"):
        yield
        return

    class _Shim:
        @staticmethod
        def tqdm(*args, **kwargs) -> _ForwardingBar:
            return _ForwardingBar(target, *args, **kwargs)

    whisper_module.tqdm = _Shim
    try:
        yield
    finally:
        whisper_module.tqdm = original
