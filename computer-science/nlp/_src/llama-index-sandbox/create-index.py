# create-index.py
import csv
from llama_index import GPTSimpleVectorIndex, SimpleWebPageReader

article_urls = []
with open('tmp/article-urls.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        article_urls.append(row[0])

documents = SimpleWebPageReader().load_data(article_urls)
index = GPTSimpleVectorIndex.from_documents(documents)
index.save_to_disk('tmp/index.json')
