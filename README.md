
# SmartLearn - AI-Powered Math Tutoring Platform

## Project Analysis Report

**SmartLearn** is an enterprise-grade educational platform that combines advanced RAG (Retrieval-Augmented Generation) technology with AI-powered exercise generation to deliver personalized mathematics tutoring. The system processes over 550 mathematical documents across multiple disciplines including Algebra, Calculus, Geometry, Trigonometry, and Statistics.

---

## 📊 Table of Contents
- [System Architecture Overview](#-system-architecture-overview)
- [Current Performance Analysis](#-current-performance-analysis)
- [Optimization Strategy](#-optimization-strategy)
- [Technical Components](#-technical-components)


---

##  Data Processing Pipeline

### 10-Step Chunking Pipeline (`chunk3.py`)

The system uses a sophisticated 10-step pipeline to process raw educational content:

```
1. INITIALISATION       → Load configs, tokenizer (tiktoken)
2. LECTURE DES FICHIERS → Read 550+ JSON files from processed3_data/
3. EXTRACTION HIÉRARCHIE → Detect course (Algebra, Calculus, etc.)
4. DÉCOUPAGE EN PHRASES → NLTK sentence tokenization
5. DÉTECTION SECTIONS   → Identify definitions, theorems, proofs, examples
6. CRÉATION CHUNKS      → Create 300-500 token chunks with overlap
7. HEADERS CONTEXTUELS  → Add course/topic/section headers
8. VALIDATION QUALITÉ   → Filter noise (min 120 tokens, max 600)
9. DÉDUPLICATION        → MD5 hash-based deduplication
10. SAUVEGARDE          → Output to all_chunks_v3.json
```

**Configuration:**
- **Min Tokens**: 120 (hard limit)
- **Target Range**: 300-500 tokens (optimal for RAG)
- **Max Tokens**: 600 (hard limit)
- **Overlap**: 60 tokens (context preservation)

**Section Types Detected:**
- `definition` - Mathematical definitions
- `theorem` - Theorems, lemmas, corollaries
- `proof` - Mathematical proofs
- `example` - Worked examples and exercises
- `remark` - Notes and observations
- `solution` - Problem solutions

**Metadata Structure:**
```json
{
  "source": "CalculusVolume1-OP_sanitized.json",
  "course": "Calculus",
  "topic": "Derivatives",
  "section": "theorem",
  "section_title": "Chain Rule",
  "chunk_id": 42,
  "text": "[Calculus - Derivatives]\n[THEOREM]\nChain Rule\n\nIf f(x)...",
  "text_clean": "If f(x) is differentiable...",
  "tokens": 387
}
```

---

##  Technical Components

### 1. RAG Chatbot (`AI_assistant`)

**Key Files:**
- `rag.py`: Core RAG implementation with metrics tracking
- `api_server.py`: Async FastAPI server with `/chat` endpoint

**RAG Chain Flow:**
```python
query → embed_query() → ChromaDB search (k=3) → format_docs() → 
→ LLM prompt → Llama 3.1 → response + metrics
```

**Prompt Template:**
```python
"""
You are a university-level mathematics tutor.

Context from textbooks:
{context}

Student's question: {question}

Instructions:
1. Answer clearly and step-by-step
2. Use proper mathematical notation
3. Reference the provided context
4. If context insufficient, say so
"""
```

**Metrics Tracked:**
- Retrieval time (ms)
- Generation time (s)
- Precision @ k
- Recall @ k
- Hit rate
- Context overlap ratio

### 2. Exercise Generator (`ExerciceGen`)

**Key Features:**
- **RAG-based**: Retrieves similar exercises as templates
- **Few-shot learning**: Uses retrieved exercises to guide generation
- **Strict JSON output**: Enforces `{question, solution, final_answer}` schema
- **SymPy verification**: Mathematical answer checking

**Difficulty Mapping:**
```python
DIFFICULTY_MAP = {
    "easy": [1, 2, 3],
    "medium": [4, 5, 6],
    "hard": [7, 8]
}
```

**Generation Flow:**
```
topic + difficulty → retrieve_exercises(k=5) → 
→ build_prompt() → LLM → parse_json() → verify_format()
```

### 3. Streamlit Frontend

**Pages:**
1. **app.py**: Dashboard with hero section, navigation cards
2. **Chatbot.py**: Chat interface with session history
3. **Exercise_Generator.py**: Exercise configuration + practice interface

**Features:**
- Session state management
- Real-time API calls (60s timeout)
- Source display with expandable sections
- Answer verification with instant feedback

---

##  Getting Started

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running
   ```bash
   # Install from https://ollama.ai
   ollama serve
   ollama pull llama3.1
   ```
3. **RAM**: Minimum 8GB (16GB recommended)
4. **GPU**: Optional (NVIDIA CUDA for faster embeddings)

### Installation

```bash
# Clone repository
cd SmartLearn

# Install dependencies
pip install streamlit fastapi uvicorn langchain langchain-ollama \
    langchain-huggingface chromadb sentence-transformers \
    torch tiktoken nltk sympy
```

### Configuration

**Environment Variables (`.env`):**
```bash
# API Ports
RAG_PORT=5002
EXERCISE_PORT=5001

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# RAG Config
RAG_RETRIEVER_K=3
RAG_DEBUG_TIMING=0
```

### Running the System

**Terminal 1: Exercise Generator API**
```powershell
cd AI/ExerciceGen
python api_server.py
```
Wait for: ` Exercise Generator API v2.0`

**Terminal 2: AI Assistant API**
```powershell
cd AI/AI_assistant
python api_server.py
```
Wait for: ` Starting RAG Chatbot API`

**Terminal 3: Streamlit Frontend**
```powershell
streamlit run SmartLearn/app.py
```

**Access:**
- Frontend: http://localhost:8501
- Exercise API Docs: http://localhost:5001/docs
- Chatbot API Docs: http://localhost:5002/docs

---

## 📡 API Documentation

### Exercise Generator API (`localhost:5001`)

#### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/health` | Service health check |
| GET | `/subjects` | List available math subjects |
| POST | `/generate` | Generate 1-10 exercises |
| POST | `/check-answer` | Verify student answer |

#### Example: Generate Exercises

**Request:**
```json
POST /generate
{
  "subject": "quadratic equations",
  "difficulty": "medium",
  "count": 3
}
```

**Response:**
```json
{
  "success": true,
  "exercises": [
    {
      "id": 1,
      "question": "Solve: x² - 5x + 6 = 0",
      "solution": "Step 1: Factor...",
      "answer": "x = 2 or x = 3",
      "difficulty": "medium",
      "subject": "quadratic equations",
      "hint": "Try factoring the quadratic"
    }
  ],
  "total": 3
}
```

### AI Chatbot API (`localhost:5002`)

#### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/health` | Service health check |
| POST | `/chat` | Send message to chatbot |
| POST | `/search` | Search knowledge base |

#### Example: Chat

**Request:**
```json
POST /chat
{
  "message": "Explain the Pythagorean theorem",
  "sessionId": "user123",
  "history": []
}
```

**Response:**
```json
{
  "success": true,
  "response": "The Pythagorean theorem states that in a right triangle...",
  "sources": [
    {
      "content": "In a right-angled triangle...",
      "metadata": {
        "course": "Geometry",
        "section": "theorem"
      }
    }
  ]
}
```

---

## 🛠 Troubleshooting

### Common Issues

#### 1. MaxRetryError / HuggingFace Connection

**Symptom**: `urllib3.exceptions.MaxRetryError` on first run

**Cause**: Downloading embedding model from HuggingFace

**Solution:**
```bash
# Wait for download to complete, or download manually:
python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

#### 2. Ollama Connection Refused

**Symptom**: `Connection refused to localhost:11434`

**Solution:**
```bash
# Start Ollama server
ollama serve

# Verify in browser
# Visit: http://localhost:11434
```

#### 3. Slow Performance

**Symptom**: Queries take >15 seconds

**Solutions:**
1. Follow [Optimization Recommendations](#-optimization-recommendations)
2. Check GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True for GPU
   ```
3. Monitor Ollama:
   ```bash
   ollama ps  # Check running models
   ```

#### 4. ChromaDB Errors

**Symptom**: `Collection not found` or embedding dimension mismatch

**Solution:**
```bash
# Rebuild ChromaDB index
cd AI/AI_assistant
python store_chroma.py
```

---

## 📊 Performance Benchmarks

### Current Baseline (Default Config)

| Configuration | Retrieval | Generation | Total |
| :--- | :--- | :--- | :--- |
| k=3, all-MiniLM-L6-v2, Llama 3.1 8B | 5.99s | 7.55s | **13.54s** |

### Projected Optimizations

| Configuration | Retrieval | Generation | Total | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| k=2, paraphrase-MiniLM-L3-v2, Llama 3.2 3B | 2.5s | 3.5s | **6.0s** | ↓ 56% |
| k=2, all-MiniLM-L3-v2, Phi3 Mini | 1.5s | 2.5s | **4.0s** | ↓ 70% |
| k=2, MiniLM-L3-v2, GPT-3.5-turbo (cloud) | 1.5s | 1.2s | **2.7s** | ↓ 80% |

---

##  License & Credits

**Project**: SmartLearn - AI Math Tutoring Platform  
**Technologies**: FastAPI, Streamlit, LangChain, Ollama, ChromaDB, Sentence Transformers  
**Models**: Meta Llama 3.1, HuggingFace all-MiniLM-L6-v2  
**Data**: 550+ educational math documents (Algebra, Calculus, Geometry, Statistics, etc.)

---

##  Next Steps

1. **Implement Recommended Optimizations**: Start with k=2 and Llama 3.2 3B
2. **Monitor Metrics**: Track context overlap to ensure quality maintained
3. **A/B Testing**: Compare model quality vs. speed tradeoffs
4. **Scale**: Consider horizontal scaling with load balancer for production
5. **Fine-tuning**: Implement LoRA fine-tuning for domain-specific improvements

**For questions or support**, refer to API documentation at `/docs` endpoints.

