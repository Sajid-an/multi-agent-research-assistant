from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")


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

    # search the web — 8 results for broader coverage
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=8,
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

    summary_prompt = f"""You are a senior research analyst conducting deep investigative research.
Analyze these web search results thoroughly for the query: "{query}"

Search Results:
{raw_content}

Instructions:
- Extract ALL specific facts: numbers, statistics, dates, names, versions, prices, percentages
- Cover multiple angles — technical details, real-world implications, expert opinions, criticisms
- Identify and explicitly flag any contradictions or disagreements between sources
- Note the credibility and recency of each source; prefer primary sources over aggregators
- Organize findings thematically — do NOT summarize source by source
- Include direct quotes where they add precision
- Do NOT use filler phrases; every sentence must carry information

Produce a comprehensive, deeply detailed research summary with no information left out."""

    summary = llm.invoke([HumanMessage(content=summary_prompt)])

    print(f"[Web Search Agent] Found {len(results)} results, summarized.")

    return {
        "web_results": [summary.content],
        "sources": state.get("sources", []) + sources,
        "next_agent": get_next(state, "web_search", "arxiv")
    }