from langgraph.runtime import Runtime

from metadata import PAIGraphState, PAIContext


def login_node(state: PAIGraphState):
    return {"login": "empty", "sensitive_data": "nothing"}