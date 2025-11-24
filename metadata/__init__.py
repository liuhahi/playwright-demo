import operator
from dataclasses import dataclass

from langchain.agents import AgentState
from langgraph.graph import MessagesState
from typing_extensions import Annotated

from metadata.user_stories import UserStories
from metadata.vulnerability import Vulnerability, Vulnerabilities


class PAIState(AgentState):
    vulnerabilities: Annotated[list[Vulnerability], operator.add]


class PAIGraphState(MessagesState):
    target: str
    sensitive_data: str
    user_stories: UserStories
    vulnerabilities: Annotated[list[Vulnerability], operator.add]


@dataclass
class PAIContext:
    workspace_path: str


__all__ = [
    "PAIState",
    "PAIGraphState",
    "PAIContext",
    "Vulnerability",
    "Vulnerabilities",
]
