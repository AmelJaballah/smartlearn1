import json
from pathlib import Path

INPUT_FILE = Path("chunked_data/all_chunks_v3.json")
OUTPUT_FILE = Path("chunked_data/all_chunks2.json")

DROP_SECTIONS = {"section"}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Original chunks: {len(chunks)}")

filtered = [
    c for c in chunks
    if c.get("section") not in DROP_SECTIONS
]

print(f"Filtered chunks: {len(filtered)}")
print(f"Removed: {len(chunks) - len(filtered)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

print("✅ Section chunks removed successfully")