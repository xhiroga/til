# LlamaIndex

```shell
op inject --force -i .env.tpl -o .env

poetry install
poetry shell
dotenv run -- python create-index.py
dotenv run -- python query.py
```

## References

- [jerryjliu/llama\_index: LlamaIndex \(GPT Index\) is a project that provides a central interface to connect your LLM's with external data\.](https://github.com/jerryjliu/llama_index)
- [LlamaIndex（GPT Index）にDevelopersIOの記事を100件読み込ませて質問してみた \| DevelopersIO](https://dev.classmethod.jp/articles/llama-index-developersio-articles/)
