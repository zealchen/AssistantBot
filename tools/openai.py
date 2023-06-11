from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from llm.llm_openai import chat_openai

@tool('openai')
def tool_openai(prompt) -> str:
    "To chat with openai llm"

    messages = [
        SystemMessage(content="You are a helpful assistant to answer all kinds of questions."),
        HumanMessage(content=prompt)
    ]
    return chat_openai(messages).content, '', ''
