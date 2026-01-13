import ijson
from pathlib import Path
from ijson.common import IncompleteJSONError
import json


# CONFIG
INPUT_JSON = Path("json_output/combined_exercises.json")
OUTPUT_JSONL = Path("json_output/combined_exercises.jsonl")

# SAFETY
if not INPUT_JSON.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_JSON}")

OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

print("🔹 Converting JSON → JSONL (streaming, safe)...")

written = 0
skipped = 0

with open(INPUT_JSON, "rb") as fin, open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    try:
        for item in ijson.items(fin, "item"):
            try:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
            except Exception:
                skipped += 1
    except IncompleteJSONError:
        print(" WARNING: JSON ended prematurely.")
        print(" Conversion stopped at last valid object.")

print("\n✅ CONVERSION COMPLETE")
print(f" Records written: {written}")
print(f" Records skipped: {skipped}")
print(f" Output JSONL: {OUTPUT_JSONL.resolve()}")