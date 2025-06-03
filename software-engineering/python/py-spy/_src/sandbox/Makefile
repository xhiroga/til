.PHONY: profile profile-detailed profile-top

TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

PROFILE_DIR := profiles

$(PROFILE_DIR):
	mkdir -p $(PROFILE_DIR)

# 基本的なプロファイリング（フレームグラフ生成）
profile: $(PROFILE_DIR)
	uv run py-spy record \
		-o $(PROFILE_DIR)/flame_$(TIMESTAMP).svg \
		-d 30 \
		-- python main.py

# 詳細なプロファイリング（高頻度サンプリング）
profile-detailed: $(PROFILE_DIR)
	uv run py-spy record \
		-o $(PROFILE_DIR)/flame_detailed_$(TIMESTAMP).svg \
		-r 1000 \
		-d 30 \
		-- python main.py

# リアルタイムプロファイリング（インタラクティブ）
profile-top:
	uv run py-spy top -- python main.py
