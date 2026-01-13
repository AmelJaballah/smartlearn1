import json
import numpy as np
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

# FORCE GPU
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA is NOT available. GPU cannot be used.")

torch.cuda.set_device(0)
device = torch.device("cuda")

print("✅ CUDA available")
print("✅ Using GPU:", torch.cuda.get_device_name(0))

# Paths
CHUNKS_FILE = Path("chunked_data/all_chunks2.json") 
OUTPUT_DIR = Path("embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

texts = [chunk["text"] for chunk in chunks]

metadata = [
    {
        "source": chunk["source"],
        "course": chunk.get("course", "Mathematics"),
        "topic": chunk.get("topic", "General"),
        "section": chunk["section"],
        "section_title": chunk.get("section_title"),
        "chunk_id": chunk["chunk_id"]
    }
    for chunk in chunks
]
'''# Load Sentence-Transformer 
model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device=device
)'''
# Load Sentence-Transformer 
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

print("Model device:", model.device)

# Generate embeddings (GPU)
embeddings = model.encode(
    texts,
    batch_size=64,              
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True   
)

print("Embeddings shape:", embeddings.shape)

# Save outputs
np.save(EMBEDDINGS_FILE, embeddings)

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("✅ Embeddings saved to:", EMBEDDINGS_FILE)
print("✅ Metadata saved to:", METADATA_FILE)

# Final GPU confirmation
print("Final GPU memory usage (MB):",
      torch.cuda.memory_allocated() // (1024 * 1024))
