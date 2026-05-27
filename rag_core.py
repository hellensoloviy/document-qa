import os
import shutil
import time
import json
import re
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

_vectorstore = None

# ─────────────────────────────────

def get_vectorstore():
    """
    Return the vectorstore, creating it if it doesn't exist yet.
    This is a lazy singleton — created once, reused after that.
    """
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

    return _vectorstore

# ── Function 1: Process and store a document ───────────────────────────────

def process_document(file_path: str) -> dict:
    """
    Load a PDF, split into chunks, store in ChromaDB.
    Returns info about what was processed.
    """
    global _vectorstore

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    total_text = " ".join([doc.page_content for doc in documents])
    if len(total_text.strip()) < 100:
        raise ValueError("PDF appears to contain no readable text — it may be image-based")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    if len(chunks) == 0:
        raise ValueError("No chunks were created from the document. File error.")

    # Clear existing data through ChromaDB's own API.
    # Never delete the folder with shutil.rmtree while the process is running —
    # ChromaDB holds the SQLite connection open and the file becomes readonly.
    if _vectorstore is not None:
        _vectorstore.delete_collection()
        _vectorstore = None

    _vectorstore = Chroma.from_documents(
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
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})

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

    answer = call_with_retry(chain, question)

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
    
    result = call_with_retry(chain, {"text": text})
    
    return {
        "summary": result,
        "status": "success"
    }

# ── Function 4: Retry helper ──────────────────────────────────────────────────

def call_with_retry(chain, input_data, max_retries: int = 3, delay: float = 1.0):
    """
    Call a LangChain chain with automatic retry on failure.
    Waits longer between each retry (exponential backoff).
    
    max_retries: how many times to try before giving up
    delay: starting wait time in seconds (doubles each retry)
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # 1s, 2s, 4s
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s... Error: {e}")
                time.sleep(wait_time)
    
    # All retries exhausted
    raise Exception(f"Failed after {max_retries} attempts. Last error: {last_error}")

# ── Function 5: Extract tasks from meeting notes ────────────────────────────

def extract_tasks(text: str) -> dict:
    """
    Extract actionable tasks from meeting notes.
    Returns structured JSON with tasks, owners, and priorities.
    """
    prompt = ChatPromptTemplate.from_template("""
    You are an expert at extracting action items from meeting notes.
    Analyze the following meeting notes and extract all actionable tasks.

    Meeting notes:
    {text}

    Return ONLY a valid JSON object in exactly this format, no other text:
    {{
        "tasks": [
            {{
                "task": "description of the task",
                "owner": "person responsible (or TBD if not mentioned)",
                "due_date": "date or deadline if mentioned (or TBD)",
                "priority": "high/medium/low based on context"
            }}
        ],
        "meeting_date": "date of meeting if mentioned (or unknown)",
        "total_tasks": number
    }}
    """)
    
    chain = prompt | llm | StrOutputParser()
    result = call_with_retry(chain, {"text": text})
    
    
    # Clean up the response in case LLM added extra text
    clean_result = re.sub(r'```json|```', '', result).strip()
    
    try:
        parsed = json.loads(clean_result)
        return {
            "tasks": parsed,
            "status": "success"
        }
    except json.JSONDecodeError:
        # If parsing fails, return the raw text
        return {
            "tasks": result,
            "status": "success",
            "note": "Could not parse as structured JSON"
        }
