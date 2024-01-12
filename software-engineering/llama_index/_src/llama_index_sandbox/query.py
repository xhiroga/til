from llama_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex.load_from_disk('tmp/index.json')
answer = index.query("DevelopersIOブログ内において、LlamaIndexに関する記事のURLを1つください。また、記事を要約してください。")
print(answer)
