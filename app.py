import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os

# Page setup
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# Header
st.title("🤖 AI Document Assistant")

st.markdown("---")

# Description
col1, col2, col3 = st.columns(3)
with col1:
    st.info("📄 Upload any PDF document")
with col2:
    st.info("🔍 AI searches for relevant content")
with col3:
    st.info("💡 Get instant accurate answers")

st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This app uses **Retrieval Augmented Generation (RAG)** to answer questions about your documents.

**Tech Stack:**
- 🦙 LLaMA 3.1 (Groq)
- 🔗 Langchain
- 🗄️ ChromaDB
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

# Main content
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type="pdf",
    help="Upload any PDF file to get started"
)

if uploaded_file and groq_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("⏳ Processing your document..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success(f"✅ Document ready! {len(pages)} pages processed into {len(chunks)} chunks.")
    st.markdown("---")

    st.markdown("### 💬 Ask a Question")
    question = st.text_input(
        "Type your question here",
        placeholder="e.g. What is the main topic of this document?"
    )

    if question:
        with st.spinner("🔍 Searching document and generating answer..."):
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant. Answer the question based ONLY on the context below.
            If the answer is not in the context, say "I don't find that information in the document."

            Context:
            {context}

            Question: {question}
            """)

            llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
            retrieved_docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
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
    st.markdown("2. **Upload a PDF** document above")
    st.markdown("3. **Ask any question** about your document")
    st.markdown("---")
    st.markdown("##### 🔑 Don't have a Groq API Key?")
    st.markdown("Get one for free at [console.groq.com](https://console.groq.com)")

# Footer
st.markdown("---")
st.markdown(
    "Built by **Mohammad Murtaza** | "
    "[GitHub](https://github.com/Murtaza-data) | "
    "Powered by RAG + LLaMA + Langchain"
)
