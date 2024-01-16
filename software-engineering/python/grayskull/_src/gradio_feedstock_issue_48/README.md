# [`gradio deploy` not works with conda-installed `gradio-script.py` · Issue #48 · conda-forge/gradio-feedstock](https://github.com/conda-forge/gradio-feedstock/issues/48)

## Reproduction

See [notebook](/issue_reproduction/evidence.ipynb).

## Render meta.yaml

```powershell
conda env create -f environment.yml
conda run -n til_grayskull_gradio_feedstock_issue_48 grayskull pypi gradio
# grayskull created `gradio/meta.yaml`.
```

## Fix linting errors

[conda-forge-webservices says](https://github.com/conda-forge/gradio-feedstock/pull/49)

> For recipe:
> The home item is expected in the about section.

So I added `home` to `about` in `gradio/meta.yaml`.

## Verification

See [notebook](/issue_reproduction/evidence.ipynb).

## References

- [Conda-ForgeにPythonパッケージを公開する手順](https://zenn.dev/geb/articles/220520_condaforgepublish)
