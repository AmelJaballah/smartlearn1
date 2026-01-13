import json
import numpy as np
from pathlib import Path

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
import torch

# =====================================================
# PATHS
# =====================================================
CHUNKS_FILE = Path("chunked_data/all_chunks2.json") 
EMBEDDINGS_FILE = Path("embeddings/embeddings.npy")
CHROMA_DIR = Path("chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(EMBEDDINGS_FILE)

if len(chunks) != len(embeddings):
    raise ValueError(f"Chunks and embeddings mismatch: {len(chunks)} != {len(embeddings)}")

print(f"Loaded {len(chunks)} chunks and embeddings")

# =====================================================
# INIT CHROMADB (RESET COLLECTION)
# =====================================================
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)

# Delete existing collection to avoid dimension mismatch
try:
    client.delete_collection("math_courses")
    print("🗑️  Deleted existing collection 'math_courses'")
except ValueError:
    pass  # Collection didn't exist

collection = client.create_collection(
    name="math_courses",
    embedding_function=None,  # IMPORTANT: we provide embeddings manually
    metadata={"hnsw:space": "cosine"},
)

# =====================================================
# PREPARE + INSERT DATA (KEEP ALIGNMENT)
# =====================================================
documents, metadatas, ids, embed_list = [], [], [], []

for i, c in enumerate(chunks):
    text = (c.get("text") or "").strip()
    if not text:
        # If you skip a chunk, also skip its embedding to keep alignment.
        continue

    documents.append(text)
    metadatas.append(
        {
            "source": c.get("source", "unknown"),
            "course": c.get("course", "unknown"),
            "topic": c.get("topic", "unknown"),
            "section": c.get("section", "unknown"),
            "section_title": c.get("section_title", "unknown"),
            "chunk_id": c.get("chunk_id", i),
            "tokens": c.get("tokens", 0)
        }
    )
    ids.append(f"chunk_{i}")  # stable per original index
    embed_list.append(embeddings[i].astype(float).tolist())

BATCH_SIZE = 1000

for start in range(0, len(documents), BATCH_SIZE):
    end = start + BATCH_SIZE
    collection.add(
        documents=documents[start:end],
        embeddings=embed_list[start:end],
        metadatas=metadatas[start:end],
        ids=ids[start:end],
    )
    print(f"Inserted {min(end, len(documents))}/{len(documents)}")

print(f"✅ Data stored in ChromaDB at: {CHROMA_DIR.resolve()}")
print("Collection count:", collection.count())

# =====================================================
# OPTIONAL QUERY TEST (USES SAME EMBEDDING MODEL)
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

question = "What is an algebraic expression?"
query_vec = embedder.encode([question], normalize_embeddings=True)

results = collection.query(
    query_embeddings=query_vec.tolist(),
    n_results=5,
    where={"section": {"$in": ["definition", "note", "remark"]}},
)

print("\n🔍 RESULTS\n")
for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    print(f"Result {i+1}")
    print("Distance:", round(dist, 4))
    print("Section:", meta.get("section"))
    print("Source:", meta.get("source"))
    print(doc[:300], "\n")