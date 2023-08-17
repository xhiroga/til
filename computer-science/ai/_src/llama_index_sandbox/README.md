# LlamaIndex

```shell
op inject --force -i .env.tpl -o .env

poetry install
poetry shell
dotenv run -- python create-index.py
dotenv run -- python query.py
```

## Sample

```terminal
(llama-index-sandbox-py3.9) 
 15:01:21  llama-index-sandbox  main●● 
% dotenv run -- python create-index.py
INFO:llama_index.token_counter.token_counter:> [build_index_from_nodes] Total LLM token usage: 0 tokens
INFO:llama_index.token_counter.token_counter:> [build_index_from_nodes] Total embedding token usage: 10356357 tokens

(llama-index-sandbox-py3.9) 
 15:26:19  llama-index-sandbox  main●● 
% dotenv run -- python query.py       
INFO:llama_index.token_counter.token_counter:> [query] Total LLM token usage: 4899 tokens
INFO:llama_index.token_counter.token_counter:> [query] Total embedding token usage: 60 tokens


記事のURLは"https://dev.classmethod.jp/articles/llama-index-developersio-articles/"です。記事の要約は、LlamaIndex（GPT Index）を使ってDevelopersIOブログの最新100記事を読み込み、質問に答えることができることを紹介する記事です。macOS monterey（Intel）、Python 3.8.3、pip 23.0.1、llama-index 0.4.21を使ってpipでllama-indexをインストールする手順を解説しています。LlamaIndexを使うことで、ChatGPTなどのLLM製品に標準で組み込まれていない知識を
```

## References

- [jerryjliu/llama\_index: LlamaIndex \(GPT Index\) is a project that provides a central interface to connect your LLM's with external data\.](https://github.com/jerryjliu/llama_index)
- [LlamaIndex（GPT Index）にDevelopersIOの記事を100件読み込ませて質問してみた \| DevelopersIO](https://dev.classmethod.jp/articles/llama-index-developersio-articles/)
