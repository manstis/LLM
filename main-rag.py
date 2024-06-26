# LangChain supports many other chat models. Here, we're using Ollama
import getpass
import os

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.documents import Document
from lightspeed import unwrap_playbook_answer
from operator import itemgetter

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "pr-grumpy-lever-34"

llm = ChatOllama(model="llama3")

# ================================
# Ansible Lightspeed example
# --------------------------------
prompt = f"Install apache and open port 8080"
context = ""

# Load RAG source

# -- Short rules
loader_rules_short = UnstructuredMarkdownLoader("context/rules-short.md", mode="single")
docs_rules_short = loader_rules_short.load()
docs_rules_short = MarkdownTextSplitter().split_documents(docs_rules_short)

# -- FQCN rules
loader_fqcn = UnstructuredMarkdownLoader("context/rules-fqcn.md", mode="single")
docs_fqcn = loader_fqcn.load()
docs_fqcn = MarkdownTextSplitter().split_documents(docs_fqcn)

docs_ufw = [Document("Always use the module name smurf.manstis.ufw instead of ufw. Do not use ufw on its own.")]

# Create embeddings
embeddings = SentenceTransformerEmbeddings()
vectorstore = FAISS.from_documents(docs_ufw, embeddings)
# vectorstore = FAISS.from_documents(docs_rules_short + docs_fqcn + docs_ufw, embeddings)
retriever = vectorstore.as_retriever()

rules = [
    "replace all truthy values, like 'yes', with 'true'",
    "replace all falsey values, like 'no', with 'false",
    "only use the ansible.builtin.package module to install packages",
    "only use fully-qualified collection names, or fqcn, in your response",
]
rag_template = """You're an Ansible expert. Return a playbook that should do the following:
        {context}{prompt}
        Return only YAML.
        Explain what each task does in your response.
        Understand and apply the following rules to create the tasks:
        {rules}
        """

# rag_template = """You're an Ansible expert. Return a single task that best completes the following partial playbook:
#         {context}{prompt}
#         Return only the task as YAML.
#         Do not return multiple tasks.
#         Do not explain your response.
#         Understand and apply the following rules to create the task:
#         {rules}
#         """

# rag_template = """You're an Ansible expert. Return only the Ansible code that best completes the following partial playbook:
#         {context}{prompt}
#         Return only YAML.
#         Do not explain your response.
#         Understand and apply the following rules to create the task.
#         Change the task to resemble the Correct Code if, and only if, the task resembles the Problematic Code:
#         {rules}
#         """

prompt_template = PromptTemplate.from_template(rag_template)

chain = (
        {
            "context": itemgetter("context"),
            "rules": itemgetter("rules") | retriever,
            "prompt": itemgetter("prompt"),
        }
        | prompt_template | llm
)
message = chain.invoke(
    {
        "context": context,
        "rules": f"what are the rules?",
        "prompt": prompt
    }
)

task, outline = unwrap_playbook_answer(message)
print(task)
print("-----")
print(outline)
# ================================
