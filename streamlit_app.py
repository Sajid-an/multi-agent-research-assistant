import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("Multi-Agent Research Assistant")
st.caption("Powered by LangGraph · Web Search · ArXiv · PDF Analysis · Fact Checking")

# ── Sidebar — PDF upload ─────────────────────────────────────────
with st.sidebar:
    st.header("Upload Documents")
    st.caption("Upload PDFs to include in your research")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf",
        accept_multiple_files=False
    )

    if uploaded_file:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Indexing..."):
                response = requests.post(
                    f"{API_URL}/upload/",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                )
            if response.status_code == 200:
                st.success(f"Indexed: {uploaded_file.name}")
            else:
                st.error(f"Failed: {response.json().get('detail', 'Unknown error')}")

    st.divider()

    # show indexed files
    st.subheader("Indexed Documents")
    try:
        files_response = requests.get(f"{API_URL}/files/")
        if files_response.status_code == 200:
            data = files_response.json()
            files = data.get("uploaded_files", [])
            if files:
                for f in files:
                    st.markdown(f"📄 {f}")
                st.caption(f"{len(files)} document(s) indexed")

                if st.button("Clear All Documents", use_container_width=True):
                    requests.delete(f"{API_URL}/clear/")
                    st.rerun()
            else:
                st.caption("No documents uploaded yet")
    except:
        st.caption("API not reachable")

    st.divider()
    st.caption("Tips:")
    st.caption("• Upload papers before asking document questions")
    st.caption("• Ask 'summarize my uploaded papers' for doc-only mode")
    st.caption("• Academic queries also search ArXiv automatically")

# ── Main area ────────────────────────────────────────────────────

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # render report as markdown
            st.markdown(message["content"])

            # show metadata if available
            if "metadata" in message:
                meta = message["metadata"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Agents: {' · '.join(meta.get('agents_used', []))}")
                with col2:
                    st.caption(f"Query type: {meta.get('query_type', 'unknown')}")
                with col3:
                    st.caption(f"Sources: {len(meta.get('sources', []))}")

                # show sources in expander
                if meta.get("sources"):
                    with st.expander("View all sources"):
                        for i, src in enumerate(meta["sources"], 1):
                            st.markdown(f"{i}. [{src}]({src})")
        else:
            st.markdown(message["content"])

# ── Query input ───────────────────────────────────────────────────
query = st.chat_input("Ask a research question...")

if query:
    # add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # run research
    with st.chat_message("assistant"):
        # progress tracking
        progress = st.empty()
        status_bar = st.progress(0)

        stages = [
            (0.1,  "Orchestrator planning research strategy..."),
            (0.3,  "Web Search Agent searching the internet..."),
            (0.5,  "ArXiv Agent fetching academic papers..."),
            (0.7,  "Fact Check Agent verifying claims..."),
            (0.85, "Synthesizer writing final report..."),
            (0.95, "Finalizing report..."),
        ]

        # animate progress while waiting for API
        import threading

        result_container = {"result": None, "error": None}

        def call_api():
            try:
                response = requests.post(
                    f"{API_URL}/research/",
                    json={"query": query},
                    timeout=180
                )
                if response.status_code == 200:
                    result_container["result"] = response.json()
                else:
                    result_container["error"] = response.json().get("detail", "Unknown error")
            except Exception as e:
                result_container["error"] = str(e)

        # start API call in background thread
        thread = threading.Thread(target=call_api)
        thread.start()

        # animate progress while API runs
        stage_idx = 0
        while thread.is_alive():
            if stage_idx < len(stages):
                pct, msg = stages[stage_idx]
                status_bar.progress(pct)
                progress.caption(f"⟳ {msg}")
                stage_idx += 1
            time.sleep(6)

        thread.join()
        status_bar.progress(1.0)
        progress.empty()
        status_bar.empty()

        # display result
        if result_container["error"]:
            st.error(f"Research failed: {result_container['error']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {result_container['error']}"
            })
        else:
            data = result_container["result"]
            report = data["report"]

            # render the report
            st.markdown(report)

            # show metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Agents: {' · '.join(data.get('agents_used', []))}")
            with col2:
                st.caption(f"Query type: {data.get('query_type', 'unknown')}")
            with col3:
                st.caption(f"Sources: {len(data.get('sources', []))}")

            # show sources
            if data.get("sources"):
                with st.expander("View all sources"):
                    for i, src in enumerate(data["sources"], 1):
                        st.markdown(f"{i}. [{src}]({src})")

            # save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": report,
                "metadata": {
                    "agents_used": data.get("agents_used", []),
                    "query_type": data.get("query_type", ""),
                    "sources": data.get("sources", [])
                }
            })