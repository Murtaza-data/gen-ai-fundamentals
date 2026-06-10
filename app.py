import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import networkx as nx
import tempfile
import os

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
    st.info("📄 Upload any PDF document")
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
    st.sidebar.info("**v2 Hybrid Search:** ChromaDB + BM25 combined. Finds chunks by both meaning AND exact keywords.")
else:
    st.sidebar.info("**v3 GraphRAG:** Extracts entities and relationships. Understands connections between concepts.")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**Tech Stack:**
- 🦙 LLaMA 3.3-70b (Groq)
- 🔗 Langchain
- 🗄️ ChromaDB / BM25 / GraphRAG
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

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type="pdf",
    help="Upload any PDF file to get started"
)

if uploaded_file and groq_api_key:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

    with st.spinner("⏳ Processing your document..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # ChromaDB vectorstore — used by all 3 versions
        vectorstore = Chroma.from_documents(chunks, embeddings)

    st.success(f"✅ Document ready! {len(pages)} pages, {len(chunks)} chunks.")

    # ── v3 only: Build knowledge graph ────────────────
    if "v3" in version:
        graph = nx.Graph()
        chunk_entities = {}

        with st.spinner("🕸️ Building knowledge graph from document..."):
            for i, chunk in enumerate(chunks[:20]):
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
        with st.spinner("🔍 Searching document and generating answer..."):

            # ── v1: Semantic search only ───────────────
            if "v1" in version:
                retrieved_docs = vectorstore.similarity_search(question, k=3)
                context = "\n".join([doc.page_content for doc in retrieved_docs])

            # ── v2: Hybrid search ──────────────────────
                     
            elif "v2" in version:
                # ChromaDB semantic search
                chroma_docs = vectorstore.similarity_search(question, k=3)

                # Simple keyword search — no library needed
                question_words = set(question.lower().split())
                stopwords = {"what", "is", "the", "a", "an", "in", "of",
                             "and", "or", "for", "to", "was", "are", "were",
                             "did", "do", "does", "how", "why", "when", "where"}
                keywords = question_words - stopwords

                scored = []
                for chunk in chunks:
                    text = chunk.page_content.lower()
                    score = sum(1 for word in keywords if word in text)
                    scored.append((score, chunk))
                scored.sort(key=lambda x: x[0], reverse=True)
                keyword_docs = [chunk for score, chunk in scored[:3] if score > 0]

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
                Question: {question}"""
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
                    if chunk_id < len(chunks):
                        graph_context += chunks[chunk_id].page_content + "\n\n"

                # Get semantic context
                semantic_docs = vectorstore.similarity_search(question, k=3)
                semantic_context = "\n".join([doc.page_content for doc in semantic_docs])

                # Combine both
                context = graph_context + semantic_context
                retrieved_docs = semantic_docs

            # ── Generate answer ────────────────────────
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
        st.markdown("---")

        with st.expander("🔍 View retrieved document chunks"):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")

elif uploaded_file and not groq_api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to continue.")

else:
    st.markdown("### 👆 Get Started")
    st.markdown("1. Enter your **Groq API Key** in the sidebar")
    st.markdown("2. **Select a RAG version** to try")
    st.markdown("3. **Upload a PDF** document above")
    st.markdown("4. **Ask any question** about your document")
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
