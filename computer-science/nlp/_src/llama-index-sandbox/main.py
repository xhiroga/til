import os
os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex.from_documents(documents)
