import json
import re
from pathlib import Path

# SANITIZATION FUNCTIONS

def remove_cid_artifacts(text: str) -> str:
    """Remove PDF font artifacts like (cid:123)"""
    return re.sub(r"\(cid:\d+\)", " ", text)


def normalize_math_symbols(text: str) -> str:
    """Normalize math symbols to safe ASCII/Unicode"""
    replacements = {
        "−": "-",
        "×": "*",
        "÷": "/",
        "∗": "*",
        "·": "*",
        "∕": "/",
        "≠": "!=",
        "≤": "<=",
        "≥": ">=",
        "≈": "~",
        "π": "pi",
        "√": "sqrt",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def remove_broken_math(text: str) -> str:
    """Remove obviously corrupted math expressions"""
    text = re.sub(r"[\*\=\-\+\/]{5,}", " ", text)
    text = re.sub(r"\([^\)]{40,}\)", " ", text)
    return text


def sanitize_text(text: str) -> str:
    """Apply all math sanitization steps"""
    text = remove_cid_artifacts(text)
    text = normalize_math_symbols(text)
    text = remove_broken_math(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



# PROCESS JSON FILES IN FOLDER

def process_json_file(input_path: Path, output_dir: Path):
    """Sanitize a single JSON file"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "text" not in data:
        print(f"⚠️ Skipped (no text field): {input_path.name}")
        return

    original_text = data["text"]
    sanitized_text = sanitize_text(original_text)

    data["text"] = sanitized_text
    data["sanitized_length"] = len(sanitized_text)

    output_path = output_dir / input_path.name.replace("_processed", "_sanitized")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Sanitized: {output_path.name}")


def main():
    input_dir = Path("processed2_data")
    output_dir = Path("processed3_data")
    output_dir.mkdir(exist_ok=True)

    if not input_dir.exists():
        print("❌ Folder 'processed2_data' not found")
        return

    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print("❌ No JSON files found in processed2_data")
        return

    print(f"Found {len(json_files)} JSON files\n")

    for json_file in json_files:
        try:
            process_json_file(json_file, output_dir)
        except Exception as e:
            print(f"❌ Failed {json_file.name}: {e}")

    print("\n✅ All JSON files sanitized successfully")
    print(f"📁 Output folder: {output_dir}")


if __name__ == "__main__":
    main()