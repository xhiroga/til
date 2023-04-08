from llama_index import GPTSimpleVectorIndex, RssReader
from llama_index.node_parser import SimpleNodeParser

rss_urls = ["https://aws.amazon.com/jp/blogs/news/feed/", "https://aws.amazon.com/jp/about-aws/whats-new/recent/feed/"]

documents = RssReader().load_data(rss_urls)

parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)
print(nodes)

index = GPTSimpleVectorIndex.from_documents(documents)
index.save_to_disk('tmp/index-rss.json')

from llama_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex.load_from_disk('tmp/index-rss.json')
answer = index.query("直近のAWSのブログでセキュリティに関するものを教えて下さい。")
print(answer)
