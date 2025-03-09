REMOTE_EXTRA_MODEL_PATHS := /til/software-engineering/openmanus/_src/sandbox/OpenManus/config/config.yaml

.PHONY: 

run:
	uv run OpenManus/main.py

upload:
	aws ssm put-parameter \
		--name $(REMOTE_EXTRA_MODEL_PATHS) \
		--tier "Advanced" \
		--type SecureString \
		--value file://OpenManus/config/config.yaml \
		--overwrite \
		--region ap-northeast-1

.venv:
	uv sync

OpenManus:
	git -C $@ pull || git clone https://github.com/mannaandpoem/OpenManus $@

OpenManus/config/config.yaml: OpenManus
		aws ssm get-parameter \
		--name $(REMOTE_EXTRA_MODEL_PATHS) \
		--output text \
		--with-decryption \
		--region ap-northeast-1 \
		--query 'Parameter.Value' > $@
