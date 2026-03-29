from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from agents.pdf_agent import vector_store
import os
import json
from agents.pdf_agent import vector_store as vs



llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

pdfs_available = vs is not None

def orchestrator_node(state: dict) -> dict:
    query = state["query"]
    print(f"[Orchestrator] Analyzing query: {query}")

    # check if any PDFs have been uploaded
    from agents.pdf_agent import vector_store as vs
    pdfs_available = vs is not None

    # ask LLM to classify the query and decide which agents are needed
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
1. Query contains "my document", "uploaded", "my paper", "my pdf", "summarize this/these"
   → ["pdf"] only — never add web_search or arxiv for document-only queries

2. Query is purely academic/technical (ML, AI, physics, biology, engineering, math)
   → ["web_search", "arxiv", "fact_check"]

3. Query is news/current events ("latest", "news", "recent", "today", "2025", "2026")
   → ["web_search", "fact_check"]

4. Query is general knowledge or mixed
   → ["web_search", "fact_check"]

5. Never include "pdf" unless pdfs_available is True
6. Never include "fact_check" alone — always pair with web_search or arxiv
7. Always end with "fact_check" when web_search or arxiv are included
8. Keep agent list ordered: web_search → arxiv → pdf → fact_check"""

    response = llm.invoke([HumanMessage(content=plan_prompt)])

    # parse the JSON plan
    try:
        # strip any accidental markdown formatting
        raw = response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
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

    # set the first agent to run
    first_agent = agents[0] if agents else "synthesizer"

    return {
        "next_agent": first_agent,
        "planned_agents": agents,
        "query_type": query_type,
    }