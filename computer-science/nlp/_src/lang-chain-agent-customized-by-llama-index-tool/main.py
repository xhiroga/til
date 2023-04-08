from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, SimpleWebPageReader, LLMPredictor, OpenAIEmbedding
from langchain.agents import initialize_agent, Tool
from langchain.tools.python.tool import PythonREPLTool
from langchain.chains.conversation.memory import ConversationBufferMemory

transcribe_index = GPTSimpleVectorIndex.from_documents(SimpleWebPageReader(html_to_text=True).load_data(["https://dev.classmethod.jp/articles/reintro-managed-ml-transcribe/"]))
translate_index = GPTSimpleVectorIndex.from_documents(SimpleWebPageReader(html_to_text=True).load_data(["https://dev.classmethod.jp/articles/reintro-managed-ml-translate/"]))

transcribe_index_tool = Tool(
    name="Transcribe Index", description="Amazon Transcribeについて特徴や料金を調べる際に利用することができます。", func=transcribe_index.query
)
translate_index_tool = Tool(
    name="Translate Index", description="Amazon Translateについて特徴や料金を調べる際に利用することができます。", func=translate_index.query
)

agent_executor = initialize_agent(
    tools=[
        PythonREPLTool(), transcribe_index_tool, translate_index_tool
    ], llm=OpenAI(temperature=0), agent="zero-shot-react-description", verbose=True
)

print(type(agent_executor))
print(type(agent_executor.agent))

answer = agent_executor.run("Pythonでフィボナッチ数列を計算し、10番目の数字を教えてください。")
print(answer)

answer = agent_executor.run("Amazon Transcribeの料金について要約し、日本語で回答してください。")
print(answer)

answer = agent_executor.run("Amazon Translateの料金について要約し、日本語で回答してください。")
print(answer)
