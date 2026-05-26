import os
import shutil
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Shared setup ───────────────────────────────────────────────────────────

CHROMA_PATH = "./chroma_db"
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# ── Function 1: Process and store a document ───────────────────────────────

def process_document(file_path: str) -> dict:
    """
    Load a PDF, split into chunks, store in ChromaDB.
    Returns info about what was processed.
    """
    # Load
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Check if PDF actually contains readable text
    total_text = " ".join([doc.page_content for doc in documents])
    if len(total_text.strip()) < 100:
        raise ValueError("PDF appears to contain no readable text — it may be image-based")
    
    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    # Check if chunks are not empty
    if len(chunks) == 0:
        raise ValueError("No chunks were created from the document. File error.")
    
    # Clear existing collection before storing new document
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return {
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "status": "success"
    }


# ── Function 2: Answer a question ──────────────────────────────────────────

def answer_question(question: str) -> dict:
    """
    Retrieve relevant chunks from ChromaDB and answer the question.
    """
    # Load existing vectorstore
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If the answer is not in the context, say "I don't have enough information in the provided documents to answer this question."

Context:
{context}

Question: {question}
""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = chain.invoke(question)
    
    return {
        "question": question,
        "answer": answer,
        "status": "success"
    }


# ── Function 3: Summarize meeting notes ────────────────────────────────────

def summarize_meeting(text: str) -> dict:
    """
    Take raw meeting notes or transcript and return structured summary.
    """
    prompt = ChatPromptTemplate.from_template("""
You are an expert at summarizing meetings. 
Analyze the following meeting notes and provide a structured summary.

Meeting notes:
{text}

Provide your response in this exact format:

SUMMARY:
[2-3 sentence overview of the meeting]

KEY DECISIONS:
- [decision 1]
- [decision 2]

ACTION ITEMS:
- [action item] — Owner: [person if mentioned, otherwise "TBD"]

NEXT STEPS:
- [next step 1]
""")
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"text": text})
    
    return {
        "summary": result,
        "status": "success"
    }