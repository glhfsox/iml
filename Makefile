# Common commands. Set SERVER to the ssh host of the training box (see ~/.ssh/config).
SERVER ?= server
REMOTE_DIR ?= ~/iml

.PHONY: install test lint smoke sync pull-runs mlflow-ui mlflow-up mlflow-down mlflow-status

install:            ## create .venv with all deps (HIP/ROCm torch on the linux server, PyPI torch on macOS)
	uv sync --group dev --group data

test:               ## fast tests (config, metrics, data contract); models are checked by GPU runs
	uv run pytest

lint:
	uv run ruff check src tests app notebooks tg

DATA ?= $(IML_DATA)
smoke:              ## DATA=<dataset>: two short runs on a contract dataset, then compare them
	uv run iml-train --config app/models/configs/resnet18.json --data $(DATA) --set epochs=3 --set pretrained=false --member smoke --topic data
	uv run iml-train --config app/models/configs/convnext_tiny.json --data $(DATA) --set epochs=3 --set pretrained=false --member smoke --topic data
	uv run iml-compare runs/smoke/data

sync:               ## push code + configs to the server (no data, no runs, no venv)
	rsync -az --delete --exclude .venv --exclude runs --exclude data --exclude mlruns.db --exclude .git ./ $(SERVER):$(REMOTE_DIR)/

pull-runs:          ## fetch run folders (json/csv/png + weights) from the server
	rsync -az $(SERVER):$(REMOTE_DIR)/runs/ runs/

# glhfsox's MLflow: store, artifacts, supervisor and its state all live in notebooks/glhfsox/mlflow/.
# Artifacts are uploaded through the server (mlflow-artifacts:/ -> mlartifacts/), so notebooks in any
# working directory store them in the same place. Same args as notebooks/glhfsox/mlflow/supervise.sh.
MLFLOW_DIR = notebooks/glhfsox/mlflow
MLFLOW_CMD = cd $(MLFLOW_DIR) && uv run mlflow server --backend-store-uri sqlite:///mlruns.db --artifacts-destination ./mlartifacts --host 0.0.0.0 --port 5000

mlflow-ui:          ## on the server: start MLflow on sqlite (foreground); locally: ssh -L 5000:localhost:5000 $(SERVER)
	$(MLFLOW_CMD)

mlflow-up:          ## always-on MLflow: start the supervisor (restarts the server on exit) unless running
	@$(MLFLOW_DIR)/supervise.sh up

mlflow-down:        ## stop the supervisor and the server
	@$(MLFLOW_DIR)/supervise.sh down

mlflow-status:      ## supervisor pid, /health, log path ($(MLFLOW_DIR)/state/mlflow.log)
	@$(MLFLOW_DIR)/supervise.sh status

# model artifacts (configs, ONNX export, freeze into app/models/speaker_verification/<model>/):
include app/models/Makefile
