from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from agents.web_search import get_next

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

# embeddings only loaded when first PDF is added — not at import
_embeddings = None
vector_store = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("[PDF Agent] Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def add_pdf(pdf_path: str):
    global vector_store
    print(f"[PDF Agent] Indexing: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        new_store = FAISS.from_documents(chunks, embeddings)
        vector_store.merge_from(new_store)

    print(f"[PDF Agent] Indexed {len(chunks)} chunks from {pdf_path}")

def pdf_node(state: dict) -> dict:
    query = state["query"]
    print(f"[PDF Agent] Searching uploaded documents for: {query}")

    if vector_store is None:
        print("[PDF Agent] No PDFs uploaded yet, skipping.")
        return {
            "pdf_results": ["No documents uploaded by user."],
            "next_agent": get_next(state, "pdf", "fact_check")
        }

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    docs = retriever.invoke(query)

    if not docs:
        return {
            "pdf_results": ["No relevant content found in uploaded documents."],
            "next_agent": get_next(state, "pdf", "fact_check")
        }

    context = "\n\n".join([
        f"From: {doc.metadata.get('source', 'uploaded document')} "
        f"(page {doc.metadata.get('page', '?')})\n{doc.page_content}"
        for doc in docs
    ])

    summary_prompt = f"""You are a meticulous document analyst. Thoroughly answer this query using ONLY the provided document excerpts: "{query}"

Document Excerpts:
{context}

Instructions:
- Answer exhaustively — extract every piece of relevant information from the excerpts
- Cite every claim with (document name, page X) — never make a claim without a citation
- If summarizing: identify all major themes, arguments, methodologies, results, and conclusions
- If answering a question: pull every relevant passage, even partially related ones
- Preserve exact technical terms, numbers, model names, and formulas as written
- Note explicitly if the documents partially address the query or have gaps
- Do NOT add any outside knowledge — only what is in the excerpts
- Structure your response with clear headings for each theme or section

Produce a comprehensive, fully-cited analysis."""

    summary = llm.invoke([HumanMessage(content=summary_prompt)])

    sources = list(set([
        f"{doc.metadata.get('source', 'uploaded')} p.{doc.metadata.get('page', '?')}"
        for doc in docs
    ]))

    print(f"[PDF Agent] Retrieved {len(docs)} chunks, summarized.")

    return {
        "pdf_results": [summary.content],
        "sources": state.get("sources", []) + sources,
        "next_agent": get_next(state, "pdf", "fact_check")
    }