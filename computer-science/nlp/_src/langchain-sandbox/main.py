from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)


llm = OpenAI(temperature=0.9)
print(llm("しりとりで使いたいので、「り」から始まって「り」で終わる言葉をいくつか挙げてください。"))
