import wave
from pathlib import Path

from telegram_asr_dataset.local_import import _chunk_pcm16_wav


def _silent_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\0\0" * round(seconds * 16_000))


def test_chunking_keeps_short_tail_with_previous_chunk(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _silent_wav(source, 41.0)
    chunks = _chunk_pcm16_wav(source, tmp_path / "chunks", 20.0)
    assert len(chunks) == 2
    with wave.open(str(chunks[-1]), "rb") as last:
        assert last.getnframes() == 21 * 16_000


def test_chunking_keeps_useful_tail_as_a_chunk(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _silent_wav(source, 45.0)
    chunks = _chunk_pcm16_wav(source, tmp_path / "chunks", 20.0)
    assert len(chunks) == 3
