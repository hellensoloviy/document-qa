# Document Q&A — RAG System

A retrieval-augmented generation (RAG) system that answers questions about your documents using AI.

## What it does
- Load any PDF document
- Ask questions about it in natural language
- Get accurate answers based only on document content
- Correctly says "I don't know" for out-of-scope questions

## Tech Stack
- **LangChain** — RAG pipeline orchestration
- **ChromaDB** — local vector database
- **OpenAI** — embeddings and chat completion
- **Python 3.11**

## How it works
1. Document is split into chunks (500 characters with 50 character overlap)
2. Each chunk is converted to a vector embedding via OpenAI
3. Vectors are stored in ChromaDB locally
4. On each question, the most relevant chunks are retrieved
5. Retrieved chunks + question are sent to GPT-3.5-turbo
6. Answer is generated based only on document content

## Setup

1. Clone the repo
2. Create virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install langchain langchain-openai langchain-community langchain-chroma langchain-text-splitters chromadb pypdf python-dotenv
```
4. Create `.env` file with your OpenAI API key
5. Add your PDF as `test_document.pdf`
6. Run
```bash
python rag.py
```

## Key learnings
- Tested chunk sizes from 50 to 1000 characters — smaller chunks improve retrieval precision but reduce answer completeness
- Prompt constraints prevent hallucination by restricting answers to document context only