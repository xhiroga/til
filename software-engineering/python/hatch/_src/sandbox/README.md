# Sandbox

[![PyPI - Version](https://img.shields.io/pypi/v/sandbox.svg)](https://pypi.org/project/sandbox)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sandbox.svg)](https://pypi.org/project/sandbox)

-----

## Initialized by

```sh
uvx hatch new sandbox
```

## Build and see inside

```console
$ uvx hatch clean && rm -rf dist
$ uvx hatch build

$ unzip -l dist/sandbox-0.0.1-py3-none-any.whl
Archive:  dist/sandbox-0.0.1-py3-none-any.whl
  Length      Date    Time    Name
---------  ---------- -----   ----
      156  02-02-2020 00:00   sandbox/__about__.py
      134  02-02-2020 00:00   sandbox/__init__.py
     1220  02-02-2020 00:00   sandbox-0.0.1.dist-info/METADATA
       87  02-02-2020 00:00   sandbox-0.0.1.dist-info/WHEEL
     1126  02-02-2020 00:00   sandbox-0.0.1.dist-info/licenses/LICENSE.txt
      458  02-02-2020 00:00   sandbox-0.0.1.dist-info/RECORD
---------                     -------
     3181                     6 files

$ tar -tzf dist/sandbox-0.0.1.tar.gz
sandbox-0.0.1/src/sandbox/__about__.py
sandbox-0.0.1/src/sandbox/__init__.py
sandbox-0.0.1/tests/__init__.py
sandbox-0.0.1/.gitignore
sandbox-0.0.1/LICENSE.txt
sandbox-0.0.1/README.md
sandbox-0.0.1/pyproject.toml
sandbox-0.0.1/PKG-INFO
```

### Distribution formats

- **Wheel** (`.whl`): バイナリ配布形式。インストールが高速
- **Source Distribution (sdist)** (`.tar.gz`): ソース配布形式。ソースコードを含む

### Inspecting build artifacts and contents

```sh
# Extract and examine
tar -xzf dist/sandbox-*.tar.gz -C dist/
ls -la dist/sandbox-*/

# Extract and examine
unzip dist/sandbox-*.whl -d dist/wheel_contents/
ls -la dist/wheel_contents/
```
