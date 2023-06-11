from langchain.tools import tool
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from llm.llm_openai import chat_openai


@tool('learn_new_words')
def tool_learn_new_words(words=[], lang="", examples_number=1) -> str:
    "chat with openai"

    prompt = f'Please show me the meaning of the following words in english, they are {" ".join(words)}. \
               Additionally, each word shall have a {lang} phonetic symbols accordingly. \
               And create {examples_number} sentencs in {lang} and each sentence has all these words in it.'

    messages = [
        SystemMessage(content="You are a helpful language teacher to help user to learn new words."),
        HumanMessage(content=prompt)
    ]
    return chat_openai(messages).content, '', ''
