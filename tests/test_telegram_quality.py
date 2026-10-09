from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from telegram_asr_dataset.audio import inspect_wav


def write_signal(path: Path, seconds: float, with_pauses: bool = False) -> None:
    sample_rate = 16_000
    samples = bytearray()
    for index in range(round(sample_rate * seconds)):
        in_pause = with_pauses and (index // (sample_rate // 2)) % 2 == 0
        value = 0 if in_pause else int(9_000 * math.sin(2 * math.pi * 220 * index / sample_rate))
        samples.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(samples))


class QualityTests(unittest.TestCase):
    def test_rejects_silence_and_too_short_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silence = root / "silence.wav"
            short = root / "short.wav"
            write_signal(silence, 3.0, with_pauses=True)
            # Overwrite with real digital silence.
            with wave.open(str(silence), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(b"\x00\x00" * 48_000)
            write_signal(short, 1.0)
            self.assertEqual(inspect_wav(silence).quality_decision, "reject")
            self.assertIn("near_silent", inspect_wav(silence).quality_reasons)
            self.assertEqual(inspect_wav(short).quality_decision, "reject")
            self.assertIn("too_short", inspect_wav(short).quality_reasons)

    def test_accepts_clear_active_audio_with_pauses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "clear.wav"
            write_signal(audio, 4.0, with_pauses=True)
            metrics = inspect_wav(audio)
            self.assertEqual(metrics.quality_decision, "accept")
            self.assertGreater(metrics.active_audio_ms, 1_000)
            self.assertGreaterEqual(metrics.quality_score, 80)


if __name__ == "__main__":
    unittest.main()
