import json
from langchain.tools import tool
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from llm.llm_openai import chat_openai


def format_few_shots(few_shots):
    formatted = ""
    for item in few_shots:
        formatted += f'\ninput:"{item["input"]}", output is:"{json.dumps(item["output"])}"\n'
    return formatted


@tool('input_parse')
def tool_input_parse(prompt, few_shots) -> str:
    "Parse user input to get all the key words that is nessasary to be used by a specific tool"

    messages = [
        SystemMessage(content=f"You are a helpful JSON formatter. Here are some few_shots on how to format a json output:{format_few_shots(few_shots)} Only output the JSON formatted response. If you cannot extract value from the user input, use \"\" to set the value instead."),
        HumanMessage(content=f'input: "{prompt}", output is:')
    ]
    return json.loads(chat_openai(messages).content), '', ''
