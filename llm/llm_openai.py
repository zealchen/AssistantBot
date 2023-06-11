from langchain.chat_models import ChatOpenAI
from config import OPENAI_KEY

chat_openai = ChatOpenAI(
    openai_api_key=OPENAI_KEY,
    temperature=0,
    model_name='gpt-3.5-turbo'
)
