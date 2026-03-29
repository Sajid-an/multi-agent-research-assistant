import arxiv
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from agents.web_search import get_next
import os

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

def arxiv_node(state: dict) -> dict:
    query = state["query"]
    print(f"[ArXiv Agent] Searching papers for: {query}")

    try:
        time.sleep(3)  # respect ArXiv rate limit

        client = arxiv.Client(
            page_size=8,
            delay_seconds=3,
            num_retries=2
        )
        search = arxiv.Search(
            query=query,
            max_results=8,
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

        summary_prompt = f"""You are a principal research scientist conducting a rigorous literature review.
Deeply analyze these ArXiv papers for the query: "{query}"

Papers:
{raw_content}

Instructions:
- For each paper, extract: core contribution, methodology, datasets used, key results with exact numbers, limitations
- Compare methodologies across papers — what approaches are dominant vs emerging?
- Identify where papers agree (consensus) and where they conflict (open problems)
- Note the trajectory: how has the field evolved across the publication dates?
- Highlight SOTA claims and the specific benchmarks/metrics behind them
- Flag any papers that directly contradict each other and explain the discrepancy
- Cite every specific claim with (Author et al., YYYY) format
- Be technically precise — include architecture names, hyperparameters, dataset splits where mentioned

Produce an in-depth academic literature synthesis, not a surface-level summary."""

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