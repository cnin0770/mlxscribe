from __future__ import annotations

from pathlib import Path


def transcribe_audio(
    audio_path: Path,
    *,
    model: str,
    language: str | None,
    task: str,
    word_timestamps: bool,
    temperature: float,
    condition_on_previous_text: bool,
    verbose: bool,
) -> dict:
    import mlx_whisper

    return mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language=language,
        task=task,
        word_timestamps=word_timestamps,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
        verbose=verbose,
    )
