import pdfplumber
import json
import re
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF with error handling"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"⚠️ Error reading PDF {pdf_path}: {e}")
        return ""


def clean_text(text: str) -> str:
    """Clean text while preserving math symbols"""
    # Remove headers/footers
    text = re.sub(r"^.*?(Chapter|Section)\s+\d+.*?$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove page numbers
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|\s*\d+\s*\|", "", text)
    text = re.sub(r"- \d+ -", "", text)
    
    # Remove copyright
    text = re.sub(r"©.*?\d{4}", "", text)
    text = re.sub(r"Copyright.*?\d{4}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"All rights reserved", "", text, flags=re.IGNORECASE)
    
    # Remove URLs and emails
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    
    # Remove references [1], (2)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    
    # Remove footer patterns
    text = re.sub(r"\bISBN\b.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDOI\b.*", "", text, flags=re.IGNORECASE)
    
    # Fix excessive newlines - replace multiple newlines with single space
    text = re.sub(r"\n+", " ", text)
    
    # Normalize whitespace - remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def process_pdf(pdf_path: Path, output_dir: Path) -> dict | None:
    """Process a single PDF and save it immediately"""
    print(f"Processing: {pdf_path.name}...")

    raw = extract_text_from_pdf(str(pdf_path))

    if not raw.strip():
        print(f"⚠️ Skipped (empty or unreadable): {pdf_path.name}")
        return None

    # Clean the text
    cleaned = clean_text(raw)
    
    result = {
        "filename": pdf_path.name,
        "text": cleaned,
        "raw_length": len(raw),
        "cleaned_length": len(cleaned)
    }

    output_file = output_dir / f"{pdf_path.stem}_processed.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved: {output_file.name}\n")
    return result


def main():
    """Process all PDFs"""
    pdf_folder = Path("data_pdfs")
    output_dir = Path("processed2_data")
    output_dir.mkdir(exist_ok=True)

    if not pdf_folder.exists():
        print("❌ Folder 'data_pdfs' not found")
        return

    pdfs = list(pdf_folder.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs\n")

    results = []
    skipped = []

    for pdf in pdfs:
        try:
            data = process_pdf(pdf, output_dir)
            if data is not None:
                results.append(data)
            else:
                skipped.append(pdf.name)
        except Exception as e:
            print(f"❌ Failed: {pdf.name} - {e}\n")
            skipped.append(pdf.name)

    print("\n" + "=" * 60)
    print(f"✓ Processed: {len(results)} PDFs")
    print(f"⚠️ Skipped: {len(skipped)} PDFs")

    if skipped:
        print("\nSkipped files:")
        for name in skipped:
            print(f"  - {name}")

    print(f"\n✓ All files saved in: {output_dir}/")


if __name__ == "__main__":
    main()