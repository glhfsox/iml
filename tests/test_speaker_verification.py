from telegram_asr_dataset.speaker_verification import classify_similarities


def test_high_consistent_scores_are_likely_match() -> None:
    result = classify_similarities([0.72, 0.74, 0.76], threshold=0.65)
    assert result.decision == "auto_likely_match"


def test_borderline_scores_are_uncertain() -> None:
    result = classify_similarities([0.56, 0.66], threshold=0.65)
    assert result.decision == "uncertain"


def test_low_scores_are_likely_mismatch() -> None:
    result = classify_similarities([0.40, 0.42], threshold=0.65)
    assert result.decision == "auto_likely_mismatch"


def test_no_speech_is_not_accepted() -> None:
    result = classify_similarities([], threshold=0.65)
    assert result.decision == "insufficient_speech"
