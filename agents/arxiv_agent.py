import arxiv
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from agents.web_search import get_next
import os

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

def arxiv_node(state: dict) -> dict:
    query = state["query"]
    print(f"[ArXiv Agent] Searching papers for: {query}")

    try:
        time.sleep(3)  # respect ArXiv rate limit

        client = arxiv.Client(
            page_size=5,
            delay_seconds=3,  # built-in delay between requests
            num_retries=2
        )
        search = arxiv.Search(
            query=query,
            max_results=5,  # keep this low
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = list(client.results(search))

        if not papers:
            print("[ArXiv Agent] No papers found.")
            return {
                "arxiv_results": ["No relevant papers found on ArXiv."],
                "next_agent": get_next(state, "arxiv", "fact_check")
            }

        raw_content = "\n\n".join([
            f"Title: {p.title}\n"
            f"Authors: {', '.join(a.name for a in p.authors[:3])}\n"
            f"Published: {p.published.strftime('%Y-%m-%d')}\n"
            f"Abstract: {p.summary}\n"
            f"URL: {p.entry_id}"
            for p in papers
        ])

        summary_prompt = f"""You are an expert academic researcher. Analyze these ArXiv papers 
and extract the most relevant academic findings for this query: "{query}"

Papers:
{raw_content}

Instructions:
- Focus on methodology, results, and benchmarks
- Extract specific metrics, dataset names, and model architectures
- Note publication dates — prefer recent papers (2023-2026)
- Identify consensus findings vs contradictory results
- Highlight breakthrough results or state-of-the-art claims
- Cite paper titles and authors for specific findings
- Be technical and precise

Provide a structured academic summary."""

        summary = llm.invoke([HumanMessage(content=summary_prompt)])
        sources = [p.entry_id for p in papers]

        print(f"[ArXiv Agent] Found {len(papers)} papers, summarized.")

        return {
            "arxiv_results": [summary.content],
            "sources": state.get("sources", []) + sources,
            "next_agent": get_next(state, "arxiv", "fact_check")
        }

    except Exception as e:
        print(f"[ArXiv Agent] Error: {e} — skipping ArXiv")
        return {
            "arxiv_results": ["ArXiv temporarily unavailable — results based on web search only."],
            "next_agent": get_next(state, "arxiv", "fact_check")
        }