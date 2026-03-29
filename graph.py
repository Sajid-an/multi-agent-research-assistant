from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os

# ── State definition ──────────────────────────────────────────────
# This is the shared memory all agents read from and write to

load_dotenv()

# import real agents as they get built
from agents.web_search import web_search_node
from agents.arxiv_agent import arxiv_node
from agents.pdf_agent import pdf_node
from agents.fact_checker import fact_check_node
from agents.synthesizer import synthesizer_node
from agents.orchestrator import orchestrator_node


class ResearchState(TypedDict):
    query: str                          # original user question
    messages: Annotated[List[BaseMessage], add_messages]
    web_results: List[str]              # findings from web search agent
    arxiv_results: List[str]            # findings from arxiv agent
    pdf_results: List[str]              # findings from pdf/rag agent
    fact_check_results: List[str]       # verified claims from fact checker
    final_report: str                   # synthesized output
    sources: List[str]                  # all citation URLs/references
    next_agent: str  
    planned_agents: List[str]     
    query_type: str 

# ── Agent node functions (stubs for now) ──────────────────────────
# Each returns a partial state update — only the keys it changes






# ── Router function ───────────────────────────────────────────────
# Reads next_agent from state and returns the node name to go to

def router(state: ResearchState) -> str:
    return state["next_agent"]

# ── Build the graph ───────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ResearchState)

    # add all nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("web_search",   web_search_node)
    graph.add_node("arxiv",        arxiv_node)
    graph.add_node("pdf",          pdf_node)
    graph.add_node("fact_check",   fact_check_node)
    graph.add_node("synthesizer",  synthesizer_node)

    # entry point
    graph.set_entry_point("orchestrator")

    # all possible destinations for each node
    all_routes = {
        "web_search":  "web_search",
        "arxiv":       "arxiv",
        "pdf":         "pdf",
        "fact_check":  "fact_check",
        "synthesizer": "synthesizer",
        "end":         END,
    }

    graph.add_conditional_edges("orchestrator", router, all_routes)
    graph.add_conditional_edges("web_search",   router, all_routes)
    graph.add_conditional_edges("arxiv",        router, all_routes)
    graph.add_conditional_edges("pdf",          router, all_routes)
    graph.add_conditional_edges("fact_check",   router, all_routes)
    graph.add_conditional_edges("synthesizer",  router, all_routes)

    return graph.compile()

# ── Run it ────────────────────────────────────────────────────────
from agents.pdf_agent import add_pdf
#add_pdf(r"C:\Users\shank\Downloads\Attention_mechanisms_in_computer_vision_A_survey.pdf")

research_graph = build_graph()

if __name__ == "__main__":
    initial_state = {
        "query": "LATESTS NEWS ON COMPUTER VISION",
        "messages": [],
        "web_results": [],
        "arxiv_results": [],
        "pdf_results": [],
        "fact_check_results": [],
        "final_report": "",
        "sources": [],
        "next_agent": "",
        "planned_agents": [],
        "query_type": "",
    }
    result = research_graph.invoke(initial_state)
    print("\n" + "="*50)
    print(result["final_report"])