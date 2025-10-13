.PHONY: run

CHECKPOINTS_DIR := checkpoints
CHECKPOINTS_TARBALL := $(CHECKPOINTS_DIR)/cnndm.tar.gz
# Public access is not permitted NOW !!!
CHECKPOINTS_URL := https://modelrelease.blob.core.windows.net/mass/cnndm.tar.gz

MASS:
	@rm -rf MASS microsoft-MASS-cda9f59
	wget -O - https://github.com/microsoft/MASS/tarball/cda9f59 | tar xz
	mv microsoft-MASS-cda9f59 MASS

$(CHECKPOINTS_DIR):
	mkdir -p $@

$(CHECKPOINTS_TARBALL): | $(CHECKPOINTS_DIR)
	wget -O $@ $(CHECKPOINTS_URL)

checkpoints/checkpoint_best.pt: $(CHECKPOINTS_TARBALL)
	tar -xzf $(CHECKPOINTS_TARBALL) --strip-components=1 -C $(CHECKPOINTS_DIR)
	touch $@

run: MASS checkpoints/checkpoint_best.pt
	uv run fairseq-inference-api.py	\
		--user-dir ./MASS/MASS-summarization/mass \
		--path checkpoints/checkpoint_best.pt \
		--beam 5 \
		--no-repeat-ngram-size 3 \
		--lenpen 1.0 \
		--task translation_mass \
		--source-lang src --target-lang tgt \
		/dicts
