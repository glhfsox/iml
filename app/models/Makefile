# Model artifacts: baseline configs, ONNX export and freezing a run into its model folder
# (a frozen folder holds only model.onnx: the app reads config + threshold from its metadata).
# Included by the root Makefile, so paths are relative to the repo root:
#   make configs                 make export RUN=runs/<member>/<arch>/<run>
#   make export-all              make freeze RUN=runs/<member>/<arch>/<run>
#   make init-models             stable pretrained model.onnx into every model folder
.PHONY: configs export export-all freeze init-models

PRETRAINED_MODELS = resnet18 efficientnetv2_s convnext_tiny resnest26d regnet_y_800mf

RUN ?= runs

export:             ## RUN=runs/<member>/<topic>/<run>: re-export best.pt -> model.onnx by hand (fit does it automatically)
	uv run iml-export --run $(RUN)

export-all:         ## export every finished iml run (metrics.json + best.pt) that has no model.onnx yet
	@for d in $$(find runs -name metrics.json -exec dirname {} \;); do \
		[ -f $$d/best.pt ] && [ ! -f $$d/model.onnx ] && { uv run iml-export --run $$d || echo "skipped $$d"; }; done; true

freeze:             ## RUN=runs/...: copy the run's model.onnx (config + threshold in its metadata) into app/models/speaker_verification/<model>/
	@dst=app/models/speaker_verification/$$(uv run python -c "import json; print(json.load(open('$(RUN)/config.json'))['model'])"); \
	{ [ -f $(RUN)/model.onnx ] || uv run iml-export --run $(RUN); } && \
	cp $(RUN)/model.onnx $$dst/model.onnx && echo "frozen $(RUN)/model.onnx into $$dst"

configs:            ## regenerate the baseline json config per model (app/models/configs/ is gitignored)
	mkdir -p app/models/configs
	uv run python -c "from iml.config import ExperimentConfig as C; from iml.models.spec import EXAMPLE_ARCH; \
	[C(model=m, member='anon', topic='baseline', **e).to_json(f'app/models/configs/{m}.json') for m, e in \
	{'resnet18': dict(lr=5e-4), 'efficientnetv2_s': dict(lr=3e-4, dropout=0.2), 'convnext_tiny': dict(optimizer='adamw', lr=2e-4, weight_decay=0.05), 'resnest26d': dict(lr=5e-4), 'regnet_y_800mf': dict(lr=5e-4), 'custom': dict(pretrained=False, arch=EXAMPLE_ARCH)}.items()]"

init-models:        ## stable starting point per model: ImageNet backbone + fresh head -> app/models/speaker_verification/<m>/model.onnx (existing graphs kept unless FORCE=1)
	@for m in $(PRETRAINED_MODELS); do \
		f=app/models/speaker_verification/$$m/model.onnx; \
		if [ -f $$f ] && [ -z "$(FORCE)" ]; then echo "keep $$f (FORCE=1 to overwrite)"; \
		else uv run iml-export --pretrained $$m --out $$f || exit 1; fi; done
