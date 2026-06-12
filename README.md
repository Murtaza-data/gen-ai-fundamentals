# 🤖 AI Document Assistant

An advanced Document Q&A application built with three versions of Retrieval Augmented Generation (RAG) — from Basic RAG to Hybrid Search to GraphRAG — with multi-document support, metadata filtering, query rewriting, and source citations.

## 🚀 Live Demo
**👉 [Try the app here](https://gen-ai-fundamentals-murtaza-document-assistant.streamlit.app/)**

## What It Does
Upload one or more PDFs and ask questions in plain English. The app finds relevant information from your documents and generates accurate, cited answers using AI. Three RAG versions are available — switch between them to see how retrieval quality improves.

## Three RAG Versions

### v1 — Basic RAG
- ChromaDB vector database for semantic similarity search
- Converts question and chunks into vectors, finds closest matches
- Simple and effective for straightforward questions

### v2 — Hybrid Search
- ChromaDB semantic search + BM25 keyword search combined
- Semantic search finds meaning-similar chunks
- BM25 ranks chunks by keyword frequency, rarity (IDF), and chunk length
- Combined and deduplicated results give better coverage than either alone

### v3 — GraphRAG
- Extracts entities and relationships from the document using the LLM
- Builds a knowledge graph (NetworkX) connecting concepts across chunks
- Traverses entity connections to retrieve information semantic search would miss
- Best for large complex documents with interconnected concepts

## Production Features

- **📂 Multi-document upload** — upload several PDFs into one searchable knowledge base
- **🏷️ Metadata filtering** — restrict search to a specific document via dropdown; the filter is applied before similarity search runs
- **🔄 Query rewriting** — an LLM rewrites vague questions into retrieval-friendly search queries before searching (the final answer still addresses the original question)
- **📌 Source citations** — every answer shows which document and page it came from
- **🛡️ Hallucination prevention** — strict grounding prompt; if the answer is not in the documents, the app says so instead of inventing one
- **⚡ Cached knowledge base** — documents are chunked and embedded once per upload (`st.cache_resource`), not on every interaction
- **⚠️ Scanned PDF detection** — image-based PDFs with no text layer are detected and reported instead of failing silently

## Tech Stack
- **LLM:** LLaMA 3.3-70b (Groq)
- **Framework:** Langchain
- **Vector Database:** ChromaDB
- **Keyword Search:** BM25 (rank_bm25)
- **Embeddings:** HuggingFace all-MiniLM-L6-v2
- **Knowledge Graph:** NetworkX
- **Frontend:** Streamlit

## Key Findings
Tested all 3 versions on the same documents:
- Hybrid Search outperforms Basic RAG on keyword-specific queries
- GraphRAG excels on large complex documents with many interconnected concepts
- For short structured documents, Hybrid Search gives the best results


## Future Improvements
- Section-aware smart chunking (split by document structure instead of character count)
- Reranking retrieved chunks before generation
- OCR support for scanned PDFs
- RAGAS evaluation pipeline

## Learning Notebooks
- `01_llms_embeddings_vectordb.ipynb` — LLMs, Embeddings, Vector Databases
- `02_rag.ipynb` — Retrieval Augmented Generation
- `03_langchain.ipynb` — Langchain Framework
