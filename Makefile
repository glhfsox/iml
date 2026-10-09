# Common commands. Set SERVER to the ssh host of the training box (see ~/.ssh/config).
SERVER ?= server
REMOTE_DIR ?= ~/iml
START ?= 2025-10-01
END ?= 2026-10-07
CHATS ?= 3
TARGET ?= speaker_01

.PHONY: install test lint smoke data-doctor data-rank data-pilot data-prepare sync pull-runs mlflow-ui export export-all freeze configs

install:            ## create .venv with all deps (HIP/ROCm torch on the linux server, PyPI torch on macOS)
	uv sync --group dev

test:               ## unit + smoke tests
	uv run pytest

lint:
	uv run ruff check src tests app notebooks

smoke:              ## end-to-end on synthetic data, then compare the runs
	uv run iml-train --config configs/simple_cnn.json --dummy --set epochs=3 --member smoke --topic dummy
	uv run iml-train --config configs/cnn10.json --dummy --set epochs=3 --member smoke --topic dummy
	uv run iml-compare runs/smoke/dummy

data-doctor:        ## check the local Telegram/FFmpeg collection dependencies
	uv run --group data voice-dataset doctor

data-rank:          ## count eligible voice messages without downloading them
	uv run --group data voice-dataset rank-chats --start $(START) --end $(END)

data-pilot:         ## interactively select team chats and collect the first 20 messages
	uv run --group data voice-dataset collect-telegram --start $(START) --end $(END) --expected-chat-count $(CHATS) --limit 20

data-prepare:       ## TARGET=speaker_01: validate reviewed rows and build training splits
	uv run iml-prepare-data --metadata data/metadata.csv --source-root . --target-speaker $(TARGET) --output data/processed/$(TARGET)

sync:               ## push code + configs to the server (no data, no runs, no venv)
	rsync -az --delete --exclude .venv --exclude runs --exclude data --exclude mlruns.db --exclude .git ./ $(SERVER):$(REMOTE_DIR)/

pull-runs:          ## fetch run folders (json/csv/png + weights) from the server
	rsync -az $(SERVER):$(REMOTE_DIR)/runs/ runs/

mlflow-ui:          ## on the server: start MLflow on sqlite; locally: ssh -L 5000:localhost:5000 $(SERVER)
	uv run mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000

# --- model artifacts -------------------------------------------------------------------
RUN ?= runs

export:             ## RUN=runs/<member>/<topic>/<run>: best.pt -> model.onnx, verified against pytorch
	uv run --group export iml-export --run $(RUN)

export-all:         ## export every finished run that has no model.onnx yet
	@for d in $$(find runs -name best.pt -exec dirname {} \;); do \
		[ -f $$d/model.onnx ] || uv run --group export iml-export --run $$d; done

freeze:             ## RUN=runs/...: copy best.pt + config.json into app/models/speaker_verification/<model>/ and export ONNX there
	@dst=app/models/speaker_verification/$$(uv run python -c "import json; print(json.load(open('$(RUN)/config.json'))['model'])"); \
	cp $(RUN)/best.pt $(RUN)/config.json $$dst/ && echo "frozen into $$dst" && \
	uv run --group export iml-export --run $$dst

configs:            ## regenerate the baseline json config per model (configs/ is gitignored)
	uv run python -c "from iml.config import ExperimentConfig as C; \
	[C(model=m, member='anon', topic='baseline', **e).to_json(f'configs/{m}.json') for m, e in \
	{'simple_cnn': {}, 'cnn10': dict(depth=4, dropout=0.2), 'resnet18': dict(lr=5e-4), 'mobilenet_v2': dict(dropout=0.2), 'ecapa_tdnn': dict(init='default')}.items()]"
