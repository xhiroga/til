# kedro

```powershell
conda env create -f environment.yml
conda activate kedro
```

```powershell
kedro new --starter=spaceflights-pandas
cd spaceflights-pandas
pip install -r requirements.txt

kedro run --nodes=preprocess_companies_node
```

## References

- [Next steps: Tutorial — kedro 0.19.2 documentation](https://docs.kedro.org/en/stable/tutorial/spaceflights_tutorial.html#)
