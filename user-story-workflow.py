import sys
import os

# Add the project root to sys.path so we can import settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings
from langgraph.constants import START, END
from langgraph.graph import StateGraph

# from agents.coordination import coordination_node
# from agents.report import report_node
# from agents.user_stories import user_stories_node
from agents.login import login_node
from metadata import PAIGraphState, PAIContext


def create_user_story():
    graph = StateGraph(state_schema=PAIGraphState, context_schema=PAIContext)
    graph.add_node("login", {"name": login_node})
    graph.add_edge(START, "login")
    # graph.add_edge("login", "navigation")
    # graph.add_node("navigation", {"navigation": "navigate"})
    graph.add_edge("login", END)
    return graph.compile()


if __name__ == "__main__":
    user_story = create_user_story()
    for mode, chunk in user_story.stream(
        {"target": "https://v4staging.scantist.io"},
        context=PAIContext(workspace_path=settings.PAI_WORKSPACE),
        stream_mode=["updates", "custom"],
    ):
        print(chunk)
