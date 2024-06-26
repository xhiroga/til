import os

from dotenv import load_dotenv
from gpt_index import GithubRepositoryReader, GPTSimpleVectorIndex

load_dotenv()

loader = GithubRepositoryReader(
    github_token=os.environ["GITHUB_TOKEN"],
    owner="xhiroga",
    repo="dotfiles",
    use_parser=False,
    verbose=True,
    concurrent_requests=1
)

docs = loader.load_data(branch='main')

for doc in docs:
    print(doc.extra_info)

index = GPTSimpleVectorIndex.from_documents(docs)
index.save_to_disk("data/something.json")
