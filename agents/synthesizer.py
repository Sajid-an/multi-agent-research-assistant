from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from datetime import datetime

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

def synthesizer_node(state: dict) -> dict:
    query = state["query"]
    planned = state.get("planned_agents", [])
    print(f"[Synthesizer] Writing final report for: {query}")

    web_results   = "\n".join(state.get("web_results", []))
    arxiv_results = "\n".join(state.get("arxiv_results", []))
    pdf_results   = "\n".join(state.get("pdf_results", []))
    fact_check    = "\n".join(state.get("fact_check_results", []))
    sources       = state.get("sources", [])

    # build only the sections that have data
    sections = ""
    if web_results and "web_search" in planned:
        sections += f"\n=== Web Research Findings ===\n{web_results}\n"
    if arxiv_results and "arxiv" in planned:
        sections += f"\n=== Academic Literature (ArXiv) ===\n{arxiv_results}\n"
    if pdf_results and "pdf" in planned:
        sections += f"\n=== Document Analysis ===\n{pdf_results}\n"
    if fact_check and "fact_check" in planned:
        sections += f"\n=== Fact-Check Assessment ===\n{fact_check}\n"

    synthesis_prompt = f"""You are a senior research analyst writing a professional research report.
Synthesize the following findings into a comprehensive, well-structured report.

Research Query: "{query}"

{sections}

Report requirements:
- Write for an intelligent, informed audience
- Lead with the most important and surprising findings
- Integrate findings across sources — don't just summarize each source separately
- Highlight agreements AND contradictions between sources
- Use specific numbers, names, and dates — avoid vague generalities
- Flag anything that is speculative or unverified
- Keep each section focused and concise — cut filler
- Do not repeat the same point in multiple sections

Structure:
# {query}
*Generated on {datetime.now().strftime("%B %d, %Y")}*

## Executive Summary
[3-4 sentences covering the single most important insight and key supporting points]

## Key Findings
[Thematically organized findings, integrating all sources. Use subheadings for themes.]

{"## Academic Perspective" if arxiv_results and "arxiv" in planned else ""}
{"[What peer-reviewed research specifically says, with paper citations]" if arxiv_results and "arxiv" in planned else ""}

{"## Document Insights" if pdf_results and "pdf" in planned else ""}
{"[Specific findings from uploaded documents with page references]" if pdf_results and "pdf" in planned else ""}

{"## Fact-Check Assessment" if fact_check and "fact_check" in planned else ""}
{"[Verified, partially verified, and unverified claims with confidence levels]" if fact_check and "fact_check" in planned else ""}

## Conclusion
[What do all these findings mean together? What is the overall picture?]

## Limitations
[Specific gaps in this research — what was not covered and why it matters]

Critical rules:
- Never fabricate citations or statistics not present in the findings
- If a section has no data write: 'No data available from this source'
- Contradictions between sources must be explicitly noted"""

    report = llm.invoke([HumanMessage(content=synthesis_prompt)])

    unique_sources = list(dict.fromkeys(sources))
    sources_section = "\n## Sources\n" + "\n".join([
        f"{i+1}. {src}" for i, src in enumerate(unique_sources)
    ]) if unique_sources else ""

    final_report = report.content + sources_section
    print("[Synthesizer] Report complete.")

    return {
        "final_report": final_report,
        "next_agent": "end"
    }