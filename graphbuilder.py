from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from state_functions import *



workflow = StateGraph(GraphState)

workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("generate_direct", generate_direct)

# workflow.add_edge(START,"retrieve")
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "direct_answer": "generate_direct",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)
workflow.add_conditional_edges(
    "generate",
    hallucination_check,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)
workflow.add_edge("generate_direct", END)

inmemory = MemorySaver()

agent = workflow.compile(checkpointer=inmemory)
