# LangChain

```shell
# Copy and set API key
cp .env.tpl .env

./.venv/bin/python main.py
```

## Note

`poetry shell`で仮想環境に入った場合、`python main.py`で実行すると、`ModuleNotFoundError: No module named 'dotenv'`というエラーが出る。

```terminal
% poetry shell

% python main.py
Traceback (most recent call last):
  File "/Users/hiroga/.ghq/github.com/xhiroga/til/computer-science/nlp/_src/langchain-sandbox/main.py", line 2, in <module>
    from dotenv import load_dotenv
ModuleNotFoundError: No module named 'dotenv'
```
