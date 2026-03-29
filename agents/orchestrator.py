from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
import json

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

# Keywords that unambiguously signal a document-only query.
# Checked in Python — no LLM involved — so routing is deterministic.
_DOC_KEYWORDS = [
    # explicit file references
    "my document", "my documents", "my pdf", "my pdfs",
    "my paper", "my papers", "my file", "my files",
    # intent phrases — catch typos and varied phrasing
    "based on my", "according to my", "using my",
    "from my", "in my", "about my",
    # uploaded/indexed references
    "uploaded", "the uploaded", "i uploaded",
    # article references
    "the document", "the pdf", "the paper",
    "from the document", "from the pdf", "from the paper",
    "in the document", "in the paper",
    # summary triggers
    "summarize this", "summarize these", "summarize the document",
    "summarize the pdf", "summarize the paper",
    "what does the document", "what does my",
]

def orchestrator_node(state: dict) -> dict:
    query = state["query"]
    print(f"[Orchestrator] Analyzing query: {query}")

    from agents.pdf_agent import vector_store as vs
    pdfs_available = vs is not None

    # ── Deterministic document-only guard ────────────────────────
    # If the user clearly references their uploaded files, bypass the
    # LLM entirely so a small model can never misroute this.
    query_lower = query.lower()
    if any(kw in query_lower for kw in _DOC_KEYWORDS):
        print("[Orchestrator] Document-only query detected — routing to pdf only.")
        return {
            "next_agent": "pdf",
            "planned_agents": ["pdf"],
            "query_type": "document",
        }

    # ── LLM routing for everything else ──────────────────────────
    plan_prompt = f"""You are a research orchestrator deciding which agents to deploy.

Query: "{query}"
PDFs available: {pdfs_available}

Agents:
- web_search: live internet search — current events, news, general knowledge, recent developments
- arxiv: academic paper search — ML, AI, science, engineering, technical research
- pdf: searches user-uploaded documents
- fact_check: verifies specific factual claims from other agents

Respond with ONLY valid JSON, no other text:
{{
    "reasoning": "one sentence explanation",
    "agents": ["agent1", "agent2"],
    "query_type": "academic|general|technical|news|document"
}}

Routing rules — follow strictly:
1. Query is purely academic/technical (ML, AI, physics, biology, engineering, math)
   → ["web_search", "arxiv", "fact_check"]

2. Query is news/current events ("latest", "news", "recent", "today", "2025", "2026")
   → ["web_search", "fact_check"]

3. Query is general knowledge or mixed
   → ["web_search", "fact_check"]

4. Never include "pdf" unless pdfs_available is True
5. Never include "fact_check" alone — always pair with web_search or arxiv
6. Always end with "fact_check" when web_search or arxiv are included
7. Keep agent list ordered: web_search → arxiv → fact_check"""

    response = llm.invoke([HumanMessage(content=plan_prompt)])

    try:
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        plan = json.loads(raw)
        agents = plan.get("agents", ["web_search", "fact_check"])
        reasoning = plan.get("reasoning", "")
        query_type = plan.get("query_type", "general")
    except Exception as e:
        print(f"[Orchestrator] Failed to parse plan, using defaults. Error: {e}")
        agents = ["web_search", "fact_check"]
        reasoning = "fallback plan"
        query_type = "general"

    print(f"[Orchestrator] Query type: {query_type}")
    print(f"[Orchestrator] Reasoning: {reasoning}")
    print(f"[Orchestrator] Plan: {agents}")

    return {
        "next_agent": agents[0] if agents else "synthesizer",
        "planned_agents": agents,
        "query_type": query_type,
    }