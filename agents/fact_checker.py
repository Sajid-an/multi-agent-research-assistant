from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
import os
from agents.web_search import get_next

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def fact_check_node(state: dict) -> dict:
    query = state["query"]
    print(f"[Fact Check Agent] Verifying findings for: {query}")

    web_results = state.get("web_results", [])
    arxiv_results = state.get("arxiv_results", [])
    pdf_results = state.get("pdf_results", [])

    all_findings = "\n\n".join([
        f"=== Web Findings ===\n{chr(10).join(web_results)}",
        f"=== Academic Findings ===\n{chr(10).join(arxiv_results)}",
        f"=== Document Findings ===\n{chr(10).join(pdf_results)}",
    ])

    # step 1 — extract specific claims worth verifying
    extract_prompt = f"""You are a fact-checking assistant.
From the following research findings, extract the 3 most specifically 
verifiable factual claims.

Research Findings:
{all_findings}

ONLY extract claims that are:
- Specific technical results (model performance, benchmark scores, accuracy numbers)
- Named research papers with specific results
- Specific named events or announcements
- Avoid market size projections, revenue forecasts, or analyst estimates
  as these vary widely between sources and cannot be definitively verified

Return ONLY a numbered list of 3 or 4 specific claims. Nothing else."""

    claims_response = llm.invoke([HumanMessage(content=extract_prompt)])
    claims_text = claims_response.content
    print(f"[Fact Check Agent] Extracted claims:\n{claims_text}")

    # step 2 — verify each claim with a targeted Tavily search
    verify_prompt_input = f"""You are verifying these claims extracted from research:
{claims_text}

Use the following verification search results to assess each claim:
"""

    verification_results = []
    for i, line in enumerate(claims_text.strip().split("\n")):
        if not line.strip():
            continue
        claim = line.strip().lstrip("123456789.-) ")
        if len(claim) < 10:
            continue

        print(f"[Fact Check Agent] Verifying: {claim[:60]}...")

        try:
            search = client.search(
                query=f"verify: {claim}",
                search_depth="advanced",
                max_results=3
            )
            snippets = "\n".join([
                f"- {r['title']}: {r['content'][:200]}"
                for r in search.get("results", [])
            ])
            verification_results.append(
                f"Claim: {claim}\nEvidence:\n{snippets}"
            )
        except Exception as e:
            verification_results.append(
                f"Claim: {claim}\nEvidence: Could not verify — {str(e)}"
            )

    # step 3 — LLM synthesizes verification findings
    verification_text = "\n\n".join(verification_results)

    final_prompt = f"""You are a rigorous fact-checking analyst. Assess these claims 
extracted from research about: "{query}"

Claims and Evidence:
{verify_prompt_input}
{verification_text}

For each claim provide a structured assessment:

**Claim [N]:** [restate the claim]
**Status:** VERIFIED | PARTIALLY VERIFIED | UNVERIFIED | CONTRADICTED
**Confidence:** HIGH | MEDIUM | LOW
**Evidence:** [what the verification search found]
**Caveat:** [any important nuance or correction needed]

Status definitions:
- VERIFIED: multiple sources directly confirm the claim
- PARTIALLY VERIFIED: claim is broadly correct but details may differ
- UNVERIFIED: insufficient evidence found either way — not a judgment of falsity
- CONTRADICTED: evidence directly disputes the claim

Be objective and precise. Do not verify claims you cannot find evidence for."""

    final_assessment = llm.invoke([HumanMessage(content=final_prompt)])

    print("[Fact Check Agent] Verification complete.")

    return {
        "fact_check_results": [final_assessment.content],
        "next_agent": get_next(state, "fact_check", "synthesizer")
    }   