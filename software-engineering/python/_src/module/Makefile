.ONESHELL:

.PHONY: clean

somepackage/dist/somepackage-0.1.0-py3-none-any.whl:
	mkdir -p somepackage
	cd somepackage
	uv init --package --no-workspace
	echo "def main() -> None: pass" > src/somepackage/main.py
	uvx hatch build 

clean:
	rm -rf somepackage/
