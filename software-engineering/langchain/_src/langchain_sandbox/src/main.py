from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()

llm = OpenAI(temperature=0.9)
print(llm("しりとりで使いたいので、「り」から始まって「り」で終わる言葉をいくつか挙げてください。"))
