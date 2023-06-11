import uuid
from typing import List, Union
from pydantic import BaseModel
from collections import namedtuple
import gradio as gr
from langchain.agents import AgentExecutor
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from agents import agent_registry
from agents.base_agent import CasualChatAget, BaseFewShotAgent
from tools import tool_lists
import config
SESSION_CONTEXTS = {}
CLICK_RESPONSE = namedtuple(
    'ClickResponse',
    ['chatbot', 'textbox_summary', 'textbox_detail', 'textbox_session', 'textbox_input'])


class SessionContext(BaseModel):
    user_messages: List[str] = []
    agent: Union[BaseFewShotAgent, None] = None

    def add_message(self, msg):
        self.user_messages.append(msg)

    def update_agent(self, agent):
        self.agent = agent


def retrieve_agent(session_context: SessionContext):
    if not session_context.agent or isinstance(session_context.agent, CasualChatAget):
        agent_index = []
        for agent_name, cls in agent_registry.items():
            for item in cls.few_shots():
                agent_index.append(
                    Document(
                        page_content=item['input'],
                        metadata={"agent_name": agent_name}
                    )
                )

        vector_store = FAISS.from_documents(agent_index, HuggingFaceEmbeddings())
        retriever = vector_store.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={"score_threshold": .1})
        relevant_agents = retriever.get_relevant_documents("".join(session_context.user_messages))

        if not relevant_agents:
            session_context.update_agent(CasualChatAget())
        else:
            session_context.update_agent(agent_registry[relevant_agents[0].metadata['agent_name']])


def action(user_msg, chat_history, session):
    global SESSION_CONTEXTS
    if not session:
        # create a session to hold the current conversation context
        chat_history = []
        session = str(uuid.uuid1())
        SESSION_CONTEXTS[session] = SessionContext()
    session_context = SESSION_CONTEXTS[session]
    session_context.add_message(user_msg)

    retrieve_agent(session_context)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=session_context.agent, tools=tool_lists, verbose=True)

    input_message = " ".join([item[0] + " " + item[1] for item in chat_history])
    input_message += user_msg
    response, summary, detail = agent_executor.run(input_message)
    chat_history.append((user_msg, response))
    return tuple(CLICK_RESPONSE(
        chatbot=chat_history, textbox_summary=summary, textbox_detail=detail,
        textbox_session=session, textbox_input=''
    ))


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=11):
            textbox_input = gr.Textbox(label="human")
        with gr.Column(scale=1):
            btn = gr.Button(value="send")

    with gr.Row():
        with gr.Box():
            with gr.Row():
                chatbot = gr.Chatbot(show_labe=True, label=config.BOT_TIPS)
            with gr.Row():
                with gr.Accordion("Summary", open=False):
                    textbox_summary = gr.Textbox(show_label=False, lines=30)
            with gr.Row():
                with gr.Accordion("Detail information", open=False):
                    textbox_detail = gr.Textbox(show_label=False, lines=50)

            with gr.Row():
                textbox_session = gr.Textbox(visible=False)

    btn.click(
        action,
        inputs=[textbox_input, chatbot, textbox_session],
        outputs=[chatbot, textbox_summary, textbox_detail, textbox_session, textbox_input]
    )


if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
