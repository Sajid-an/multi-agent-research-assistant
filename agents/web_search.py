from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")


def get_next(state: dict, current: str, fallback: str) -> str:
    planned = state.get("planned_agents", [])
    try:
        idx = planned.index(current)
        return planned[idx + 1] if idx + 1 < len(planned) else "synthesizer"
    except (ValueError, IndexError):
        return fallback
    
    
def web_search_node(state: dict) -> dict:
    query = state["query"]
    print(f"[Web Search Agent] Searching for: {query}")

    # search the web — max 5 results, full content not just snippets
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_raw_content=False,
        include_answer=True
    )

    # extract results and sources
    results = response.get("results", [])
    sources = [r["url"] for r in results]
    raw_content = "\n\n".join([
        f"Source: {r['url']}\nTitle: {r['title']}\nContent: {r['content']}"
        for r in results
    ])

    # use LLM to summarize findings relevant to the query
    summary_prompt = f"""You are an expert research analyst. Analyze these web search results 
and extract the most relevant, accurate, and recent information for this query: "{query}"

Search Results:
{raw_content}

Instructions:
- Focus on the most recent and credible sources
- Extract specific facts, numbers, dates, and named entities
- Ignore promotional or marketing content
- Flag any conflicting information between sources
- Organize findings by theme, not by source
- Be concise — no filler phrases like "according to the search results"

Provide a dense, information-rich summary."""

    summary = llm.invoke([HumanMessage(content=summary_prompt)])

    print(f"[Web Search Agent] Found {len(results)} results, summarized.")

    return {
        "web_results": [summary.content],
        "sources": state.get("sources", []) + sources,
        "next_agent": get_next(state, "web_search", "arxiv")
    }