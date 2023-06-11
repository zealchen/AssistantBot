from .input_parse_tool import tool_input_parse
from .openai import tool_openai
from .learn_new_words import tool_learn_new_words
tool_lists = [
    tool_learn_new_words,
    tool_openai,
    tool_input_parse
]
