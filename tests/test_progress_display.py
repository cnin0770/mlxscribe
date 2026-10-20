import importlib

from mlxscribe.progress_display import JobProgress, format_clock, whisper_progress

whisper_module = importlib.import_module("mlx_whisper.transcribe")


def test_format_clock_switches_to_hours() -> None:
    assert format_clock(75) == "01:15"
    assert format_clock(3725) == "1:02:05"


def test_job_progress_clamps_to_total() -> None:
    progress = JobProgress(100, desc="Transcribing", unit="s", clock=True)
    progress.advance(60)
    progress.advance(80)
    assert progress.position == 100
    progress.advance_to(40)  # never moves backwards
    assert progress.position == 100
    progress.close()


def test_whisper_progress_forwards_frames_as_seconds_and_restores_tqdm() -> None:
    original = whisper_module.tqdm
    progress = JobProgress(30, desc="Transcribing", unit="s")

    with whisper_progress(progress):
        with whisper_module.tqdm.tqdm(total=3000, unit="frames", disable=False) as bar:
            bar.update(1500)  # 15 seconds of audio
    assert progress.position == 15
    assert whisper_module.tqdm is original
    progress.close()


def test_whisper_progress_with_no_target_silences_updates() -> None:
    with whisper_progress(None):
        with whisper_module.tqdm.tqdm(total=3000) as bar:
            bar.update(1500)
    assert hasattr(whisper_module.tqdm, "tqdm")
