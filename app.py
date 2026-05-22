
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
st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📄")
st.title("📄 Document Q&A Chatbot")
st.markdown("Upload a PDF and ask questions about it!")

# API Key input
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and groq_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Reading and processing your PDF..."):
        # Load and split
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        # Embeddings and Vector DB
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        st.success(f"PDF processed! {len(pages)} pages, {len(chunks)} chunks ready.")

    # Question input
    question = st.text_input("Ask a question about your document")

    if question:
        with st.spinner("Finding answer..."):
            # RAG Pipeline
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
            response = chain.invoke({"context": context, "question": question})

            st.markdown("### Answer")
            st.write(response.content)

elif uploaded_file and not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar.")
else:
    st.info("Please upload a PDF to get started.")
