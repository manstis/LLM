# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
)

# ================================
# langchain example
# --------------------------------
# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model="llama3")

chat_template = ChatPromptTemplate.from_template("Tell me a very short joke about {topic}.")
chain = chat_template | llm | StrOutputParser()
print(chain.invoke({"topic": "Space travel"}))
# ================================
