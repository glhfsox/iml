from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import sys
import wave
from array import array
from pathlib import Path

from .models import AudioMetrics


class AudioProcessingError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    configured = os.environ.get("FFMPEG_BINARY", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise AudioProcessingError("FFMPEG_BINARY does not point to an executable file")

    system_binary = shutil.which("ffmpeg")
    if system_binary:
        return system_binary

    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - dependency/runtime specific
        raise AudioProcessingError(
            "FFmpeg is unavailable. Install project dependencies first."
        ) from exc
    if not Path(bundled).is_file():
        raise AudioProcessingError("The bundled FFmpeg executable was not found")
    return bundled


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_ffmpeg(arguments: list[str]) -> None:
    command = [
        find_ffmpeg(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioProcessingError(f"FFmpeg could not run: {exc}") from exc
    if completed.returncode != 0:
        detail = " ".join(completed.stderr.strip().split())[-500:]
        raise AudioProcessingError(f"FFmpeg failed: {detail or 'unknown error'}")


def extract_video_audio(source: Path, destination: Path) -> None:
    """Extract the video note's audio stream without a lossy re-encode."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part" + destination.suffix)
    temporary.unlink(missing_ok=True)
    common = [
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
    ]
    try:
        _run_ffmpeg([*common, "-c:a", "copy", "-f", "matroska", str(temporary)])
    except AudioProcessingError:
        temporary.unlink(missing_ok=True)
        # A lossless FLAC fallback handles unusual streams that cannot be muxed as-is.
        try:
            _run_ffmpeg([*common, "-c:a", "flac", "-f", "matroska", str(temporary)])
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise AudioProcessingError("Extracted audio is empty")
    os.replace(temporary, destination)
    try:
        os.chmod(destination, 0o600)
    except OSError:
        pass


def normalize_audio(source: Path, destination: Path, sample_rate_hz: int = 16_000) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part" + destination.suffix)
    temporary.unlink(missing_ok=True)
    try:
        _run_ffmpeg(
            [
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate_hz),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(temporary),
            ]
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise AudioProcessingError("Normalized WAV is empty")
    os.replace(temporary, destination)
    try:
        os.chmod(destination, 0o600)
    except OSError:
        pass


def inspect_wav(path: Path) -> AudioMetrics:
    digest = hashlib.sha256()
    sample_count = 0
    sum_squares = 0
    peak = 0
    clipped = 0
    silent = 0
    silence_threshold = 104  # approximately -50 dBFS for signed 16-bit PCM
    frame_size = 320  # 20 ms at 16 kHz
    frame_sum_squares = 0
    frame_sample_count = 0
    frame_levels_dbfs: list[float] = []

    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            if sample_width != 2:
                raise AudioProcessingError(
                    f"Expected 16-bit PCM but found {sample_width * 8}-bit samples"
                )
            while True:
                frames = wav.readframes(65_536)
                if not frames:
                    break
                digest.update(frames)
                values = array("h")
                values.frombytes(frames)
                if sys.byteorder != "little":
                    values.byteswap()
                for value in values:
                    absolute = abs(value)
                    sample_count += 1
                    sum_squares += value * value
                    peak = max(peak, absolute)
                    clipped += int(absolute >= 32_760)
                    silent += int(absolute <= silence_threshold)
                    frame_sum_squares += value * value
                    frame_sample_count += 1
                    if frame_sample_count == frame_size:
                        frame_rms = math.sqrt(frame_sum_squares / frame_sample_count)
                        frame_levels_dbfs.append(
                            20 * math.log10(frame_rms / 32768) if frame_rms else -120.0
                        )
                        frame_sum_squares = 0
                        frame_sample_count = 0
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioProcessingError(f"Invalid WAV output: {exc}") from exc

    duration_ms = round(frame_count * 1000 / sample_rate) if sample_rate else 0
    rms = math.sqrt(sum_squares / sample_count) if sample_count else 0.0
    rms_dbfs = 20 * math.log10(rms / 32768) if rms else -120.0
    peak_dbfs = 20 * math.log10(peak / 32768) if peak else -120.0
    clipping_ratio = clipped / sample_count if sample_count else 0.0
    silence_ratio = silent / sample_count if sample_count else 1.0

    if frame_sample_count:
        frame_rms = math.sqrt(frame_sum_squares / frame_sample_count)
        frame_levels_dbfs.append(20 * math.log10(frame_rms / 32768) if frame_rms else -120.0)

    ordered_levels = sorted(frame_levels_dbfs)
    if ordered_levels:
        noise_level = ordered_levels[int((len(ordered_levels) - 1) * 0.20)]
        signal_level = ordered_levels[int((len(ordered_levels) - 1) * 0.80)]
        activity_threshold = max(-45.0, noise_level + 8.0)
        active_frames = sum(level >= activity_threshold for level in frame_levels_dbfs)
        active_audio_ratio = active_frames / len(frame_levels_dbfs)
        active_audio_ms = min(duration_ms, active_frames * 20)
        estimated_snr_db = max(0.0, signal_level - noise_level)
    else:
        active_audio_ratio = 0.0
        active_audio_ms = 0
        estimated_snr_db = None

    flags: list[str] = []
    if sample_rate != 16_000 or channels != 1 or sample_width != 2:
        flags.append("wrong_format")
    if duration_ms == 0:
        flags.append("empty_audio")
    elif duration_ms < 2_000:
        flags.append("too_short")
    elif duration_ms < 3_000:
        flags.append("short_for_speaker_verification")
    if duration_ms > 90_000:
        flags.append("very_long")
    elif duration_ms > 45_000:
        flags.append("long_for_speaker_verification")
    if rms_dbfs < -45:
        flags.append("near_silent")
    if silence_ratio > 0.95:
        flags.append("mostly_silence")
    if active_audio_ms < 1_000 or active_audio_ratio < 0.10:
        flags.append("too_little_active_audio")
    elif active_audio_ratio < 0.25:
        flags.append("low_active_audio_ratio")
    if clipping_ratio > 0.05:
        flags.append("severely_clipped")
    elif clipping_ratio > 0.005:
        flags.append("clipped")

    # An energy estimate cannot reliably distinguish speech from music.  It is
    # therefore used only to route borderline recordings to human review.
    if estimated_snr_db is not None and active_audio_ratio < 0.95 and estimated_snr_db < 8.0:
        flags.append("low_estimated_snr")

    reject_reasons = {
        "wrong_format",
        "empty_audio",
        "too_short",
        "very_long",
        "near_silent",
        "mostly_silence",
        "too_little_active_audio",
        "severely_clipped",
    }.intersection(flags)
    review_reasons = set(flags) - reject_reasons
    quality_decision = "reject" if reject_reasons else ("review" if review_reasons else "accept")
    qc_status = (
        "fail"
        if quality_decision == "reject"
        else ("warn" if quality_decision == "review" else "pass")
    )
    deductions = {
        "wrong_format": 100,
        "empty_audio": 100,
        "too_short": 60,
        "very_long": 30,
        "near_silent": 60,
        "mostly_silence": 60,
        "too_little_active_audio": 60,
        "severely_clipped": 50,
        "short_for_speaker_verification": 15,
        "long_for_speaker_verification": 15,
        "low_active_audio_ratio": 20,
        "clipped": 20,
        "low_estimated_snr": 25,
    }
    quality_score = max(0, 100 - sum(deductions.get(flag, 0) for flag in flags))
    return AudioMetrics(
        duration_ms=duration_ms,
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bits=sample_width * 8,
        rms_dbfs=round(rms_dbfs, 3),
        peak_dbfs=round(peak_dbfs, 3),
        clipping_ratio=round(clipping_ratio, 8),
        silence_ratio=round(silence_ratio, 8),
        active_audio_ms=active_audio_ms,
        active_audio_ratio=round(active_audio_ratio, 8),
        estimated_snr_db=(round(estimated_snr_db, 3) if estimated_snr_db is not None else None),
        pcm_sha256=digest.hexdigest(),
        qc_status=qc_status,
        qc_flags=tuple(flags),
        quality_score=quality_score,
        quality_decision=quality_decision,
        quality_reasons=tuple(flags),
    )
