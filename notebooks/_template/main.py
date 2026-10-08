"""<member>'s entry point to the production code with one chosen run.

Copy this folder to notebooks/<your-name>/ and point RUN at the run you want to demo:
    uv run python notebooks/<your-name>/main.py --wav path/to/recording.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> `app` importable

from app.main import load_model  # noqa: E402

# A frozen model name (app/models/speaker_verification/<name>/ after `make freeze`)
# or a run folder runs/<member>/<topic>/<run_name>.
RUN = "simple_cnn"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=RUN)
    p.add_argument("--wav", required=True)
    args = p.parse_args()
    verdict = load_model(args.run).predict_file(args.wav)
    print(f"{verdict.name}  (p_allowed={verdict.p_allowed:.3f}, mc-std={verdict.uncertainty:.3f})")


if __name__ == "__main__":
    main()
