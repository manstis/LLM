from langchain_community.chat_models import ChatOllama
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from textwrap import dedent
from lightspeed import unwrap_playbook_answer

llm = ChatOllama(model="llama3")

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

text = "Install apache and open port 8080"

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
