import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import networkx as nx
import tempfile

# ════════════════════════════════════════════════════════
# PAGE SETUP
# ════════════════════════════════════════════════════════

st.set_page_config(page_title="AI Document Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Document Assistant")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("📄 Upload one or more PDF documents")
with col2:
    st.info("🔍 AI searches for relevant content")
with col3:
    st.info("💡 Get instant accurate answers")

st.markdown("---")

# ════════════════════════════════════════════════════════
# SIDEBAR — version choice + API key
# ════════════════════════════════════════════════════════

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### 🔢 Select RAG Version")
version = st.sidebar.radio(
    "Version",
    ["v1 — Basic RAG", "v2 — Hybrid Search", "v3 — GraphRAG"]
)

if "v1" in version:
    st.sidebar.info("**v1 Basic RAG:** ChromaDB semantic search only. Finds chunks by meaning similarity.")
elif "v2" in version:
    st.sidebar.info("**v2 Hybrid Search:** ChromaDB + BM25 keyword search combined. Finds chunks by meaning AND exact keywords.")
else:
    st.sidebar.info("**v3 GraphRAG:** Extracts entities and relationships. Understands connections between concepts.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Corrective RAG")
st.sidebar.info("After retrieval, the AI grades whether the chunks are relevant. If weak, it rewrites the query and retries. If still weak, it falls back to a web search.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Tech Stack:**
- 🦙 LLaMA 3.3-70b (Groq)
- 🔗 Langchain
- 🗄️ ChromaDB + BM25 / GraphRAG
- 🤗 HuggingFace Embeddings
- 🌐 DuckDuckGo (web fallback)
""")

st.sidebar.markdown("---")
groq_api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password",
                                     help="Get your free API key at console.groq.com")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 [GitHub Profile](https://github.com/Murtaza-data)")

# ════════════════════════════════════════════════════════
# KNOWLEDGE BASE  (the slow setup — cached so it runs ONCE)
# --------------------------------------------------------
# Loads PDFs, chunks them, embeds them, stores in ChromaDB.
# @st.cache_resource means: only re-run if the uploaded files
# change. Every other rerun reuses the saved result instantly.
# ════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_knowledge_base(files_data):
    all_pages = []
    for name, data in files_data:
        # PyPDFLoader needs a real file path, so write the bytes to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        # Stamp each page with its REAL filename (for citations + filtering)
        for page in pages:
            page.metadata["source"] = name
        all_pages.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return all_pages, chunks, vectorstore

# ════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════

uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf",
                                  accept_multiple_files=True,
                                  help="Upload one or more PDF files to get started")

if uploaded_files and groq_api_key:

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

    # files_data = the cache "ID card" (name + bytes). Must be a tuple (hashable).
    files_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
    with st.spinner("⏳ Processing your documents..."):
        all_pages, chunks, vectorstore = build_knowledge_base(files_data)

    st.success(f"✅ {len(uploaded_files)} document(s) ready! {len(all_pages)} pages, {len(chunks)} chunks.")

    # ── METADATA FILTERING ───────────────────────────────
    # Choose which document to search. "Auto" lets the LLM decide
    # later (per question); a specific file restricts the search now.
    doc_choice = st.sidebar.selectbox(
        "📂 Search in:",
        ["🤖 Auto (AI decides)", "All documents"] + [f.name for f in uploaded_files]
    )

    if doc_choice in ("🤖 Auto (AI decides)", "All documents"):
        active_chunks = chunks
        chroma_filter = None
    else:
        active_chunks = [c for c in chunks if c.metadata.get("source") == doc_choice]
        chroma_filter = {"source": doc_choice}

    # Guard: scanned/image PDFs produce no text → no chunks → stop gracefully
    if len(active_chunks) == 0:
        st.error(f"⚠️ No readable text found in {doc_choice}. It may be a scanned PDF (images, not text). Try another document.")
        st.stop()

    # ── v3 ONLY: BUILD KNOWLEDGE GRAPH ───────────────────
    # For each chunk, the LLM extracts entities; entities in the
    # same chunk get connected, with the chunk_id stored on the edge.
    graph = None
    graph_chunks = active_chunks
    if "v3" in version:
        graph = nx.Graph()
        with st.spinner("🕸️ Building knowledge graph from document..."):
            for i, chunk in enumerate(graph_chunks[:20]):
                entity_prompt = f"""Extract 3-5 key entities (people, places, concepts, organizations) from this text.
                Return ONLY a comma-separated list. Nothing else.
                Text: {chunk.page_content}"""
                response = llm.invoke([HumanMessage(content=entity_prompt)])
                entities = [e.strip() for e in response.content.split(",")]
                for entity in entities:
                    graph.add_node(entity)
                for j in range(len(entities)):
                    for k in range(j + 1, len(entities)):
                        graph.add_edge(entities[j], entities[k], chunk_id=i)
        st.success(f"🕸️ Knowledge graph built: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships")

    st.markdown("---")

    # ════════════════════════════════════════════════════
    # QUESTION & ANSWER  (with Corrective RAG)
    # ════════════════════════════════════════════════════

    st.markdown("### 💬 Ask a Question")
    question = st.text_input("Type your question here",
                             placeholder="e.g. What is the main topic of this document?")

    if question:
        with st.spinner("🔍 Searching documents and generating answer..."):

            # ── BLOCK 1: AUTO-ROUTING ────────────────────
            # In Auto mode, the LLM reads the question and picks the
            # single document most likely to answer it, then filters to it.
            routed_doc = None
            if doc_choice == "🤖 Auto (AI decides)" and len(uploaded_files) > 1:
                file_names = [f.name for f in uploaded_files]
                routing_prompt = f"""You are a document router. These documents are available: {file_names}
                Which single document is most likely to contain the answer to this question?
                Reply with ONLY the exact filename from the list, or ALL if the question could apply to any document.
                Question: {question}"""
                candidate = llm.invoke([HumanMessage(content=routing_prompt)]).content.strip()
                if candidate in file_names:
                    routed_chunks = [c for c in chunks if c.metadata.get("source") == candidate]
                    if len(routed_chunks) > 0:
                        active_chunks = routed_chunks
                        chroma_filter = {"source": candidate}
                        routed_doc = candidate

            # ── BLOCK 2: SEARCH RECIPE (the 3 versions) ──
            # One reusable function holding all 3 retrieval methods.
            # The corrective loop calls this each time it needs chunks.
            # Returns: context (text for the LLM) + docs (objects for citations).
            def retrieve_context(query):
                if "v1" in version:
                    docs = vectorstore.similarity_search(query, k=5, filter=chroma_filter)
                    return "\n".join(d.page_content for d in docs), docs

                elif "v2" in version:
                    # semantic (ChromaDB) + keyword (BM25), combined and deduplicated
                    chroma_docs = vectorstore.similarity_search(query, k=5, filter=chroma_filter)
                    bm25 = BM25Retriever.from_documents(active_chunks)
                    bm25.k = 5
                    keyword_docs = bm25.invoke(query)
                    seen, docs = set(), []
                    for d in chroma_docs + keyword_docs:
                        if d.page_content not in seen:
                            seen.add(d.page_content)
                            docs.append(d)
                    return "\n".join(d.page_content for d in docs), docs

                else:  # v3 — graph traversal + semantic search
                    q_ent = llm.invoke([HumanMessage(content=f"""Extract key entities from this question.
                    Return ONLY a comma-separated list. Nothing else.
                    Question: {query}""")]).content
                    q_entities = [e.strip() for e in q_ent.split(",")]
                    connected = set()
                    for entity in q_entities:
                        if entity in graph:
                            for neighbor in graph.neighbors(entity):
                                ed = graph.get_edge_data(entity, neighbor)
                                if ed and "chunk_id" in ed:
                                    connected.add(ed["chunk_id"])
                    graph_context = ""
                    for cid in list(connected)[:3]:
                        if cid < len(graph_chunks):
                            graph_context += graph_chunks[cid].page_content + "\n\n"
                    sem_docs = vectorstore.similarity_search(query, k=5, filter=chroma_filter)
                    sem_context = "\n".join(d.page_content for d in sem_docs)
                    return graph_context + sem_context, sem_docs

            # ── BLOCK 3: CORRECTIVE LOOP ─────────────────
            # Try up to 3 times. Attempt 1 uses the ORIGINAL question.
            # Each pass: (search) → (grade). If RELEVANT → break early.
            # If WEAK → rewrite the query differently and try again.
            max_attempts = 3
            search_query = question          # attempt 1 = original question, no rewrite
            context, retrieved_docs = "", []
            grade = "WEAK"
            attempts_log = []

            for attempt in range(max_attempts):
                # Rewrite ONLY on retries (attempt 1 keeps the original question)
                if attempt > 0:
                    rewrite_prompt = f"""Rewrite this question to be clearer and more specific for searching a document.
                    Expand abbreviations and replace vague words with specific terms.
                    This is retry number {attempt} — phrase it differently from previous attempts.
                    Return ONLY the rewritten question. Nothing else.
                    Question: {question}"""
                    search_query = llm.invoke([HumanMessage(content=rewrite_prompt)]).content.strip()

                # Search (calls the recipe above)
                context, retrieved_docs = retrieve_context(search_query)

                # Grade: does the retrieved text actually answer the question?
                grade_prompt = f"""Are these document chunks relevant AND sufficient to answer the question?
                Question: {question}
                Chunks: {context}
                Reply with ONLY one word: RELEVANT or WEAK."""
                grade = llm.invoke([HumanMessage(content=grade_prompt)]).content.strip().upper()

                attempts_log.append(f"Attempt {attempt+1}: '{search_query}' → {grade}")
                if "RELEVANT" in grade:
                    break   # good chunks found — stop retrying

            # ── BLOCK 4: DECIDE — documents vs web fallback ──
            # If the loop found relevant chunks, answer from documents.
            # If all attempts failed, search the web instead.
            used_web = False
            if "RELEVANT" in grade:
                answer_prompt = ChatPromptTemplate.from_template("""
                You are a helpful assistant. Answer the question based ONLY on the context below.
                Context:
                {context}
                Question: {question}
                """)
                final_answer = (answer_prompt | llm).invoke(
                    {"context": context, "question": question}).content
            else:
                used_web = True
                # Direct DuckDuckGo call (robust to the package rename), never crashes
                def web_search(query):
                    try:
                        from ddgs import DDGS
                    except ImportError:
                        from duckduckgo_search import DDGS
                    with DDGS() as ddgs:
                        results = ddgs.text(query, max_results=5)
                    return "\n".join(r.get("body", "") for r in results)

                try:
                    web_results = web_search(question)
                except Exception:
                    web_results = ""

                if web_results:
                    web_prompt = ChatPromptTemplate.from_template("""
                    The user's documents did NOT contain the answer. Using the web search results below,
                    give a general, helpful answer. Be clear this is general web information, not from their documents.
                    Web results:
                    {web}
                    Question: {question}
                    """)
                    final_answer = (web_prompt | llm).invoke(
                        {"web": web_results, "question": question}).content
                else:
                    final_answer = "I couldn't find this in your documents, and the web search is currently unavailable. Please try rephrasing your question."

        # ── BLOCK 5: DISPLAY ─────────────────────────────
        if used_web:
            st.warning("⚠️ I couldn't find this in your documents. Here's general information from the web:")
            st.markdown(f"> {final_answer}")
            st.caption("🌐 Source: Web search (not your documents)")
        else:
            st.markdown("### 📝 Answer")
            st.markdown(f"> {final_answer}")
            sources = sorted(set(
                f"{d.metadata.get('source','document')} — Page {d.metadata.get('page',0)+1}"
                for d in retrieved_docs))
            st.caption(f"📌 Sources: {' | '.join(sources)}")

        st.caption(f"🔄 Search query used: {search_query}")
        if routed_doc:
            st.caption(f"🤖 AI routed this question to: {routed_doc}")

        # ── BLOCK 6: EXPANDERS — show the inner workings ──
        with st.expander("🔧 Corrective RAG — retrieval attempts"):
            for log in attempts_log:
                st.text(log)

        if not used_web:
            with st.expander("🔍 View retrieved document chunks"):
                for i, doc in enumerate(retrieved_docs):
                    src = doc.metadata.get("source", "document")
                    pg = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Chunk {i+1}** — 📄 {src}, Page {pg}:")
                    st.write(doc.page_content)
                    st.markdown("---")

elif uploaded_files and not groq_api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to continue.")

else:
    st.markdown("### 👆 Get Started")
    st.markdown("1. Enter your **Groq API Key** in the sidebar")
    st.markdown("2. **Select a RAG version** to try")
    st.markdown("3. **Upload one or more PDF** documents above")
    st.markdown("4. **Ask any question** about your documents")
    st.markdown("---")
    st.markdown("##### 🔑 Don't have a Groq API Key?")
    st.markdown("Get one for free at [console.groq.com](https://console.groq.com)")

# ════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "Built by **Mohammad Murtaza** | "
    "[GitHub](https://github.com/Murtaza-data) | "
    "Powered by RAG + LLaMA + Langchain"
)
