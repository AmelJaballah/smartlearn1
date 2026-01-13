"""
FastAPI Server for RAG Chatbot
High-performance async API for math tutoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import sys
import time

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =====================================================
# Configuration
# =====================================================
# 🔥 FORCE OFFLINE MODE
# 🔥 FORCE OFFLINE MODE (Default to 1, but allow override)
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RETRIEVER_K = int(os.getenv("RAG_RETRIEVER_K", "3"))
DEBUG_TIMING = os.getenv("RAG_DEBUG_TIMING", "0") == "1"
RAG_PORT = int(os.getenv("RAG_PORT", "5002"))

# Detect GPU
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# =====================================================
# Pydantic Models
# =====================================================
class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    success: bool
    response: str
    sources: List[Dict[str, Any]] = []
    sessionId: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

class SearchResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    count: int

class HealthResponse(BaseModel):
    status: str
    service: str
    collection: str
    embedding_model: str
    embeddings_device: str
    retriever_k: int
    ollama_base_url: str
    llm: str

# =====================================================
# Initialize Components
# =====================================================
print("🔄 Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE},
)
print(f"✅ Embeddings loaded on {DEVICE}")

print("🔄 Loading ChromaDB...")
vectorstore = Chroma(
    collection_name="math_courses",
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
print("✅ ChromaDB loaded")

print(f"🔄 Setting up retriever (k={RETRIEVER_K})...")
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVER_K,
        "filter": {
            "section": {
                "$in": ["definition", "theorem", "proof", "note", "remark", "example"]
            }
        }
    }
)

print(f"🔄 Initializing Ollama LLM ({OLLAMA_MODEL})...")
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)

# Prompt template
prompt = PromptTemplate.from_template("""
You are a university-level mathematics tutor. Answer the student's question using the provided context from math textbooks.

Context from textbooks:
{context}

Student's question: {question}

Previous conversation:
{history}

Instructions:
1. Answer clearly and step-by-step
2. Use proper mathematical notation
3. Reference the provided context when relevant
4. If the context doesn't contain enough information, say so
5. Provide examples when helpful
6. Be encouraging and pedagogical

Answer:
""")

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ])

def format_history(history):
    """Format conversation history"""
    if not history:
        return "No previous conversation."
    formatted = []
    for msg in history[-3:]:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)

# Build RAG chain
rag_chain = (
    {
        "context": lambda x: format_docs(x.get("docs", [])),
        "question": lambda x: x["question"],
        "history": lambda x: format_history(x.get("history", []))
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG Chatbot API initialized\n")

# =====================================================
# FastAPI App
# =====================================================
app = FastAPI(
    title="SmartLearn RAG Chatbot",
    description="High-performance math tutoring chatbot with RAG",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="rag-chatbot",
        collection="math_courses",
        embedding_model="all-MiniLM-L6-v2",
        embeddings_device=DEVICE,
        retriever_k=RETRIEVER_K,
        ollama_base_url=OLLAMA_BASE_URL,
        llm=OLLAMA_MODEL
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG-powered math tutor"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message is required")

        t_start = time.perf_counter()

        # Retrieve documents (ONCE)
        t0 = time.perf_counter()
        docs = await retriever.ainvoke(request.message)
        t_retr = time.perf_counter() - t0

        # Generate response
        t0 = time.perf_counter()
        response = await rag_chain.ainvoke({
            "question": request.message,
            "history": request.history or [],
            "docs": docs
        })
        t_llm = time.perf_counter() - t0

        if DEBUG_TIMING:
            print(f"⏱️  retrieval={int(t_retr*1000)}ms llm={int(t_llm*1000)}ms total={int((time.perf_counter()-t_start)*1000)}ms")

        # Format sources
        sources = [
            {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]

        return ChatResponse(
            success=True,
            response=response,
            sources=sources,
            sessionId=request.sessionId
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search the knowledge base"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query is required")

        results = vectorstore.similarity_search(request.query, k=request.k)

        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get('source', 'unknown')
            }
            for doc in results
        ]

        return SearchResponse(
            success=True,
            results=formatted_results,
            count=len(formatted_results)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting RAG Chatbot API (FastAPI)")
    print("=" * 60)
    print(f"📍 URL: http://localhost:5002")
    print(f"📚 Docs: http://localhost:5002/docs")
    print(f"💾 Vector Store: Chroma (math_courses)")
    print(f"🤖 LLM: {OLLAMA_MODEL} via Ollama")
    print(f"🔧 Device: {DEVICE}")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=RAG_PORT, log_level="info")
