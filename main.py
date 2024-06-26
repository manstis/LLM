# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from textwrap import dedent
from lightspeed import unwrap_playbook_answer
from operator import itemgetter

# ================================
# langchain example
# --------------------------------
# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model="llama3")

# chat_template = ChatPromptTemplate.from_template("Tell me a very short joke about {topic}.")
# chain = chat_template | llm | StrOutputParser()
# print(chain.invoke({"topic": "Space travel"}))
# ================================


# ================================
# Ansible Lightspeed example
# --------------------------------
SYSTEM_MESSAGE_TEMPLATE = """
        You are an Ansible expert.
        Your role is to help Ansible developers write playbooks.
        You answer with an Ansible playbook.
        """

HUMAN_MESSAGE_TEMPLATE = """
        This is what the playbook should do: {text}
        """

text = f"Install apache and open port 8080"

# Load RAG source
loader = UnstructuredMarkdownLoader("context/rules-short.md", mode="single")
docs = loader.load()
md_splitter = MarkdownTextSplitter()
docs = md_splitter.split_documents(docs)

# Create embeddings
embeddings = SentenceTransformerEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            dedent(SYSTEM_MESSAGE_TEMPLATE),
            additional_kwargs={"role": "system"},
        ),
        HumanMessagePromptTemplate.from_template(
            dedent(HUMAN_MESSAGE_TEMPLATE), additional_kwargs={"role": "user"}
        ),
    ]
)

chain = chat_template | llm
message = chain.invoke({"text": text})
playbook, outline = unwrap_playbook_answer(message)
print(playbook)
print("-----")
print(outline)
# ================================
