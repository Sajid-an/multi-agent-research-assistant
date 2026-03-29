# Multi-Agent Research Assistant

A production-ready AI research system powered by a team of specialized agents that collaborate to research any topic — searching the web, fetching academic papers, analyzing uploaded documents, verifying claims, and synthesizing everything into a structured, cited report.

Built with LangGraph, FastAPI, and Streamlit.

---

## How It Works

The system uses 5 specialized agents orchestrated by LangGraph:

| Agent | Role |
|-------|------|
| Orchestrator | Analyzes the query and decides which agents to deploy |
| Web Search | Searches the live internet via Tavily API |
| ArXiv | Fetches and summarizes relevant academic papers |
| PDF / RAG | Searches user-uploaded documents using FAISS vector search |
| Fact Checker | Cross-verifies claims from all agents with confidence levels |
| Synthesizer | Merges all findings into a structured markdown report with citations |

The orchestrator is smart — it routes queries efficiently:
- News queries → Web Search + Fact Check only
- Academic queries → Web Search + ArXiv + Fact Check
- Document queries → PDF agent only
- Mixed queries → all relevant agents

---

## Features

- Multi-source research — web, ArXiv, and your own PDFs in one report
- Smart agent routing — only runs agents relevant to your query
- Fact verification — claims are cross-checked with confidence levels (VERIFIED / PARTIALLY VERIFIED / UNVERIFIED / CONTRADICTED)
- Multi-PDF support — upload multiple documents, all indexed into a single FAISS vector store
- Structured reports — executive summary, key findings, academic perspective, fact-check assessment, conclusion, limitations, and numbered citations
- Chat interface — conversational UI with full history and progress tracking
- REST API — fully documented FastAPI backend at `/docs`

---

## Tech Stack

- LangGraph — multi-agent orchestration and state management
- LangChain — LLM chains and RAG pipeline
- Groq API (LLaMA 3.1 8B) — fast, free LLM for all agents
- Tavily API — web search optimized for LLM agents
- ArXiv Python library — academic paper retrieval
- FAISS — local vector store for PDF search
- HuggingFace sentence-transformers (all-MiniLM-L6-v2) — document embeddings
- FastAPI — REST API backend
- Streamlit — chat frontend
- Docker — containerized deployment

---

## Project Structure
```
MULTI AGENTS/
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── web_search.py
│   ├── arxiv_agent.py
│   ├── pdf_agent.py
│   ├── fact_checker.py
│   └── synthesizer.py
├── graph.py
├── README.md
├── main.py
├── endpoints.py
├── streamlit_app.py
├── Dockerfile
├── requirements.txt
└── .env
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/research/` | Submit a research query, get a full report |
| POST | `/upload/` | Upload a PDF to include in research |
| GET | `/files/` | List all indexed documents |
| DELETE | `/clear/` | Clear all uploaded documents and reset index |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-research-assistant
cd multi-agent-research-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file:
```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

Get your free keys:
- Groq: https://console.groq.com
- Tavily: https://tavily.com

### 5. Run the backend
```bash
uvicorn main:app --reload
```

### 6. Run the frontend
```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`

---

## Example Queries
```
"What are the latest advances in vision transformers?"
→ runs web search + arxiv + fact check

"Latest news on OpenAI"
→ runs web search + fact check only

"Summarize my uploaded papers"
→ runs PDF agent only
```

---

## Live Demo

[https://sanji02-research-agent.hf.space](#)

---

## Author

**Sajid Ahmed Ansari**
Master of IT (AI) · Macquarie University, Sydney
[LinkedIn](https://linkedin.com/in/sajidahmedansari)