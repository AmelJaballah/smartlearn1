import json
from pathlib import Path
from typing import Any, Dict
import torch
from sentence_transformers import SentenceTransformer
import chromadb

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "json_output" / "combined_exercises.jsonl"

CHROMA_DIR = BASE_DIR / "chroma_exercises"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "exercise_bank"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128   
# UTILS
def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

# GPU SETUP

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available")

device = "cuda"
print("✅ Using GPU:", torch.cuda.get_device_name(0))

# EMBEDDING MODEL
print("🔹 Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

# CHROMA PERSISTENT CLIENT
print("🔹 Initializing ChromaDB (PERSISTENT)...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine",
        "embedding_model": EMBEDDING_MODEL  
    }
)

# INDEXATION
documents: list[str] = []
metadatas: list[Dict[str, str]] = []
ids: list[str] = []

indexed = 0
skipped = 0
batch_id = 0
missing_fields = 0

print(" Indexing FULL dataset...")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):

        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        question = safe_str(
            entry.get("clean_problem")
            or entry.get("question")
            or entry.get("text")
        )

        if not question:
            skipped += 1
            continue

        # Track missing optional fields (debug)
        if not entry.get("subject") or not entry.get("difficulty"):
            missing_fields += 1

        documents.append(question)

        metadatas.append({
            "subject": safe_str(entry.get("subject")),
            "difficulty": safe_str(entry.get("difficulty")),
            "format": safe_str(entry.get("format")),
            "source": safe_str(entry.get("source")),
            "answer": safe_str(
                entry.get("extracted_answer")
                or entry.get("normalized_answer")
                or entry.get("answer")
            ),
        })

        # 🔒 Stable ID (VERY IMPORTANT)
        ids.append(safe_str(entry.get("id") or f"ex_{line_num}"))
        indexed += 1

        # BATCH COMMIT
        if len(documents) >= BATCH_SIZE:
            embeddings = embedder.encode(
                documents,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            collection.upsert(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )

            batch_id += 1
            print(f"✔ Batch {batch_id} | Indexed {indexed}")

            documents.clear()
            metadatas.clear()
            ids.clear()

# FINAL BATCH
if documents:
    embeddings = embedder.encode(
        documents,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    collection.upsert(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )

    print(f"✔ Final batch | Indexed {indexed}")

# VERIFICATION
print("\n Verifying persistence...")
print("documents in DB:", collection.count())

print("\n✅ FULL INDEXATION COMPLETE")
print(f"Total indexed exercises: {indexed}")
print(f"⚠️ Total skipped records: {skipped}")
print(f"⚠️ Missing subject/difficulty: {missing_fields}")
print(f" ChromaDB directory: {CHROMA_DIR.resolve()}")
print(f" Collection name: {COLLECTION_NAME}")