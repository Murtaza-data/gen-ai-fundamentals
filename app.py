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

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

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
# SIDEBAR
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
    st.sidebar.info("**v2 Hybrid Search:** ChromaDB + BM25 keyword search combined. Finds chunks by both meaning AND exact keywords.")
else:
    st.sidebar.info("**v3 GraphRAG:** Extracts entities and relationships. Understands connections between concepts.")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**Tech Stack:**
- 🦙 LLaMA 3.3-70b (Groq)
- 🔗 Langchain
- 🗄️ ChromaDB + BM25 / GraphRAG
- 🤗 HuggingFace Embeddings
""")

st.sidebar.markdown("---")
groq_api_key = st.sidebar.text_input(
    "🔑 Enter Groq API Key",
    type="password",
    help="Get your free API key at console.groq.com"
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 [GitHub Profile](https://github.com/Murtaza-data)")

# ════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF files to get started"
)

if uploaded_files and groq_api_key:

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

    with st.spinner("⏳ Processing your documents..."):
        # Load all PDFs — label every page with its original filename
        all_pages = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Replace temp file path with the real filename in metadata
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            all_pages.extend(pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(all_pages)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # ChromaDB vectorstore — used by all 3 versions
        vectorstore = Chroma.from_documents(chunks, embeddings)

    st.success(f"✅ {len(uploaded_files)} document(s) ready! {len(all_pages)} pages, {len(chunks)} chunks.")

    # ── Metadata filtering — choose which document to search ──
    doc_choice = st.sidebar.selectbox(
        "📂 Search in:",
        ["All documents"] + [f.name for f in uploaded_files]
    )

    if doc_choice == "All documents":
        active_chunks = chunks
        chroma_filter = None
    else:
        active_chunks = [c for c in chunks if c.metadata.get("source") == doc_choice]
        chroma_filter = {"source": doc_choice}

    # ── v3 only: Build knowledge graph ────────────────
    if "v3" in version:
        graph = nx.Graph()
        chunk_entities = {}

        with st.spinner("🕸️ Building knowledge graph from document..."):
            for i, chunk in enumerate(active_chunks[:20]):
                entity_prompt = f"""Extract 3-5 key entities (people, places, concepts, organizations) from this text.
                Return ONLY a comma-separated list. Nothing else.
                Text: {chunk.page_content}"""

                response = llm.invoke([HumanMessage(content=entity_prompt)])
                entities = [e.strip() for e in response.content.split(",")]
                chunk_entities[i] = entities

                for entity in entities:
                    graph.add_node(entity)

                for j in range(len(entities)):
                    for k in range(j + 1, len(entities)):
                        graph.add_edge(entities[j], entities[k], chunk_id=i)

        st.success(f"🕸️ Knowledge graph built: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships")

    st.markdown("---")

    # ════════════════════════════════════════════════════
    # QUESTION & ANSWER
    # ════════════════════════════════════════════════════

    st.markdown("### 💬 Ask a Question")
    question = st.text_input(
        "Type your question here",
        placeholder="e.g. What is the main topic of this document?"
    )

    if question:
        with st.spinner("🔍 Searching documents and generating answer..."):

            # ── Query rewriting — improve the question for retrieval ──
            rewrite_prompt = f"""Rewrite this question to be clearer and more specific for searching a document.
            Expand abbreviations and replace vague words with specific terms.
            Return ONLY the rewritten question. Nothing else.
            Question: {question}"""
            rewrite_response = llm.invoke([HumanMessage(content=rewrite_prompt)])
            search_query = rewrite_response.content.strip()

            # ── v1: Semantic search only ───────────────
            if "v1" in version:
                retrieved_docs = vectorstore.similarity_search(
                    search_query, k=3, filter=chroma_filter)
                context = "\n".join([doc.page_content for doc in retrieved_docs])

            # ── v2: Hybrid search ──────────────────────
            elif "v2" in version:
                # ChromaDB semantic search (respects document filter)
                chroma_docs = vectorstore.similarity_search(
                    search_query, k=3, filter=chroma_filter)

                # BM25 keyword search (built only from the filtered chunks)
                bm25_retriever = BM25Retriever.from_documents(active_chunks)
                bm25_retriever.k = 3
                keyword_docs = bm25_retriever.invoke(search_query)

                # Combine and deduplicate
                seen = set()
                retrieved_docs = []
                for doc in chroma_docs + keyword_docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        retrieved_docs.append(doc)
                context = "\n".join([doc.page_content for doc in retrieved_docs])

            # ── v3: GraphRAG ───────────────────────────
            else:
                # Extract entities from question
                q_entity_prompt = f"""Extract key entities from this question.
                Return ONLY a comma-separated list. Nothing else.
                Question: {search_query}"""
                q_response = llm.invoke([HumanMessage(content=q_entity_prompt)])
                q_entities = [e.strip() for e in q_response.content.split(",")]

                # Find connected chunks via graph
                connected_chunk_ids = set()
                for entity in q_entities:
                    if entity in graph:
                        for neighbor in graph.neighbors(entity):
                            edge_data = graph.get_edge_data(entity, neighbor)
                            if edge_data and "chunk_id" in edge_data:
                                connected_chunk_ids.add(edge_data["chunk_id"])

                # Get graph context
                graph_context = ""
                for chunk_id in list(connected_chunk_ids)[:3]:
                    if chunk_id < len(active_chunks):
                        graph_context += active_chunks[chunk_id].page_content + "\n\n"

                # Get semantic context (respects document filter)
                semantic_docs = vectorstore.similarity_search(
                    search_query, k=3, filter=chroma_filter)
                semantic_context = "\n".join([doc.page_content for doc in semantic_docs])

                # Combine both
                context = graph_context + semantic_context
                retrieved_docs = semantic_docs

            # ── Generate answer — uses the ORIGINAL question ──
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant. Answer the question based ONLY on the context below.
            If the answer is not in the context, say "I don't find that information in the document."

            Context:
            {context}

            Question: {question}
            """)

            chain = prompt | llm
            response = chain.invoke({
                "context": context,
                "question": question
            })

        st.markdown("### 📝 Answer")
        st.markdown(f"> {response.content}")

        # Show sources — filename + page
        sources = sorted(set(
            f"{doc.metadata.get('source', 'document')} — Page {doc.metadata.get('page', 0) + 1}"
            for doc in retrieved_docs
        ))
        st.caption(f"📌 Sources: {' | '.join(sources)}")
        st.caption(f"🔄 Search query used: {search_query}")

        st.markdown("---")

        with st.expander("🔍 View retrieved document chunks"):
            for i, doc in enumerate(retrieved_docs):
                source_name = doc.metadata.get("source", "document")
                page_num = doc.metadata.get("page", 0) + 1
                st.markdown(f"**Chunk {i+1}** — 📄 {source_name}, Page {page_num}:")
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
