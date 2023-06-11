from enum import Enum
from abc import abstractmethod
from typing import List, Tuple, Any, Union
from langchain.agents import BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from agents import agent_registry


class BaseFewShotAgent(BaseSingleActionAgent):
    """Agent to debug deployment fail reason"""
    
    @property
    def input_keys(self):
        return ["input"]

    @staticmethod
    @abstractmethod
    def few_shots():
        return []


class CasualChatAget(BaseFewShotAgent):
    """casual chat with human"""
    @staticmethod
    def few_shots():
        return [
            'hi',
            'hello'
        ]
    
    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if not intermediate_steps:
            return AgentAction(tool="openai", tool_input=kwargs["input"], log="")
        else:
            return AgentFinish(return_values={'output': intermediate_steps[0][1]}, log='')

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        raise NotImplementedError("does not support async")


@agent_registry.register('language_study')
class LanguageStudyAgent(BaseFewShotAgent):
    """Help you learn a new language."""
    class State(Enum):
        parse = 0
        process = 1
        finish = 2


    @staticmethod
    def few_shots():
        return [
            {
                "input": "To learn some new words: serendipity, discrepency. Target language is English. Need 3 sentences.",
                "output": {"lang": "Chinese", "words": "serendipity, discrepency", "examples_number": 3}
            },
            {
                "input": "Help me create 5 examples based on the words: 天, 地. I want to study Chinese.",
                "output": {"lang": "Chinese", "words": "天, 地", "examples_number": 5}
            }
        ]

    def state(self, intermediate_steps):
        if len(intermediate_steps) == 0:
            return LanguageStudyAgent.State.parse
        elif len(intermediate_steps) == 1:
            return LanguageStudyAgent.State.process
        else:
            return LanguageStudyAgent.State.finish
    
    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if self.state(intermediate_steps) == LanguageStudyAgent.State.parse:
            return AgentAction(tool="input_parse", tool_input={
                'prompt': kwargs["input"],
                'few_shots': self.few_shots()
            }, log="")
        elif self.state(intermediate_steps) == LanguageStudyAgent.State.process:
            info, _, _ = intermediate_steps[0][1]
            if not info.get('lang'):
                tips = 'What language do you want to learn?'
                return AgentFinish(return_values={'output': (tips, '', '')}, log='')
            if not info.get('words'):
                tips = 'What words do you want to learn?'
                return AgentFinish(return_values={'output': (tips, '', '')}, log='')
            if not info.get('examples_number'):
                tips = 'How many example of sentences to create?'
                return AgentFinish(return_values={'output': (tips, '', '')}, log='')
            return AgentAction(tool="learn_new_words", tool_input=info, log="")
        else:
            ret = intermediate_steps[1][1]
            return AgentFinish(return_values={'output': ret}, log='')

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        raise NotImplementedError("does not support async")
