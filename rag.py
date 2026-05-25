import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── STEP 1: Load the document ──────────────────────────────────────────────
print("Loading document...")

# loader = PyPDFLoader("test_document.pdf")
loader = TextLoader("test_document.txt")

documents = loader.load()
print(f"Loaded {len(documents)} page(s)")

# ── STEP 2: Split into chunks ───────────────────────────────────────────────
print("Splitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ── STEP 3: Create embeddings and store in ChromaDB ────────────────────────
print("Creating embeddings and storing in ChromaDB...")

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Stored in ChromaDB successfully")

# ── STEP 4: Set up retrieval chain ─────────────────────────────────────────
print("Setting up question answering...")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
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

# ── STEP 5: Ask your question ──────────────────────────────────────────────
# question = "What is this document about?"
question = "How does John Clute describes Wells?"
# question = "What is the full name of the author of the book Harry Potter?" Cheking for hallucinations here

print(f"\nQuestion: {question}")
print("Asking AI...")

result = chain.invoke(question)

print(f"\nAnswer: {result}")