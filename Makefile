# Common commands. Set SERVER to the ssh host of the training box (see ~/.ssh/config).
SERVER ?= server
REMOTE_DIR ?= ~/iml

.PHONY: install test lint smoke sync pull-runs mlflow-ui export export-all freeze configs

install:            ## create .venv with all deps (HIP/ROCm torch on the linux server, PyPI torch on macOS)
	uv sync --group dev --group data

test:               ## unit + smoke tests
	uv run pytest

lint:
	uv run ruff check src tests app notebooks

DATA ?= $(IML_DATA)
smoke:              ## DATA=<dataset>: two short runs on a contract dataset, then compare them
	uv run iml-train --config configs/simple_cnn.json --data $(DATA) --set epochs=3 --member smoke --topic data
	uv run iml-train --config configs/cnn10.json --data $(DATA) --set epochs=3 --member smoke --topic data
	uv run iml-compare runs/smoke/data

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
	mkdir -p configs
	uv run python -c "from iml.config import ExperimentConfig as C; \
	[C(model=m, member='anon', topic='baseline', **e).to_json(f'configs/{m}.json') for m, e in \
	{'simple_cnn': {}, 'cnn10': dict(depth=4, dropout=0.2), 'resnet18': dict(lr=5e-4), 'mobilenet_v2': dict(dropout=0.2), 'ecapa_tdnn': dict(init='default')}.items()]"
