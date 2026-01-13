"""
Enhanced RAG Chunking System - chunk3.py
10-Step Pipeline for High-Quality Educational Content Chunking

Steps:
1. Initialisation
2. Lecture des fichiers  
3. Extraction de la hiérarchie
4. Découpage en phrases
5. Détection des sections
6. Création des chunks
7. Ajout des headers contextuels
8. Validation de qualité
9. Déduplication
10. Sauvegarde et analyse
"""

import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import nltk
import tiktoken

# =====================================================
# STEP 1: INITIALISATION
# =====================================================

print("=" * 60)
print("STEP 1: INITIALISATION")
print("=" * 60)

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# Configuration
CONFIG = {
    "input_dir": Path("processed3_data"),
    "output_dir": Path("chunked_data"),
    "min_tokens": 120,
    "target_min_tokens": 300,
    "target_max_tokens": 500,
    "max_tokens": 600,
    "overlap_tokens": 60,
}

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Create output directory
CONFIG["output_dir"].mkdir(exist_ok=True)

print(f"✅ Input directory: {CONFIG['input_dir']}")
print(f"✅ Output directory: {CONFIG['output_dir']}")
print(f"✅ Token range: {CONFIG['min_tokens']}-{CONFIG['max_tokens']}")
print(f"✅ Target range: {CONFIG['target_min_tokens']}-{CONFIG['target_max_tokens']}")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(tokenizer.encode(text))


# =====================================================
# STEP 3: EXTRACTION DE LA HIÉRARCHIE
# =====================================================

# Course detection patterns
COURSE_PATTERNS = {
    "Algebra": [
        r"algebra", r"polynomial", r"equation", r"linear\s+algebra",
        r"matrix", r"matrices", r"vector\s+space", r"eigenvalue",
        r"quadratic", r"factoring", r"binomial"
    ],
    "Calculus": [
        r"calculus", r"derivative", r"integral", r"differentiation",
        r"integration", r"limit", r"continuity", r"differential",
        r"antiderivative", r"series", r"sequence"
    ],
    "Geometry": [
        r"geometry", r"triangle", r"circle", r"polygon", r"angle",
        r"congruent", r"similar", r"perpendicular", r"parallel",
        r"euclidean", r"plane", r"solid geometry"
    ],
    "Trigonometry": [
        r"trigonometry", r"sine", r"cosine", r"tangent", r"radian",
        r"periodic", r"amplitude", r"phase"
    ],
    "Statistics": [
        r"statistics", r"probability", r"distribution", r"mean",
        r"variance", r"standard\s+deviation", r"regression",
        r"hypothesis", r"sampling", r"random\s+variable"
    ],
    "Number Theory": [
        r"number\s+theory", r"prime", r"divisibility", r"modular",
        r"congruence", r"diophantine", r"integer"
    ],
    "Analysis": [
        r"analysis", r"real\s+analysis", r"complex\s+analysis",
        r"topology", r"metric\s+space", r"convergence"
    ],
    "Discrete Mathematics": [
        r"discrete", r"combinatorics", r"graph\s+theory", r"set\s+theory",
        r"logic", r"boolean", r"recursion"
    ],
}

# Topic extraction patterns
TOPIC_PATTERNS = [
    (r"chapter\s*\d*[:\s]*([A-Za-z\s]+)", 1),
    (r"section\s*\d*[:\s]*([A-Za-z\s]+)", 1),
    (r"unit\s*\d*[:\s]*([A-Za-z\s]+)", 1),
    (r"(?:lesson|topic)[:\s]+([A-Za-z\s]+)", 1),
]


def detect_course(text: str, filename: str) -> str:
    """Detect the course type from content and filename."""
    combined = f"{filename.lower()} {text[:5000].lower()}"
    
    scores = {}
    for course, patterns in COURSE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, combined, re.IGNORECASE))
            score += matches
        if score > 0:
            scores[course] = score
    
    if scores:
        return max(scores, key=scores.get)
    return "Mathematics"


def detect_topic(text: str, section_type: str, section_title: str) -> str:
    """Detect the topic from content."""
    # Try to extract from section title first
    if section_title and len(section_title) > 3:
        # Clean up section title
        topic = re.sub(r"^(definition|theorem|lemma|proof|example|exercise)[:\s]*", 
                      "", section_title, flags=re.IGNORECASE)
        if len(topic) > 3:
            return topic.strip().title()
    
    # Try to extract from text patterns
    text_sample = text[:2000]
    for pattern, group in TOPIC_PATTERNS:
        match = re.search(pattern, text_sample, re.IGNORECASE)
        if match:
            topic = match.group(group).strip()
            if len(topic) > 3 and len(topic) < 50:
                return topic.title()
    
    return "General"


# =====================================================
# STEP 4: DÉCOUPAGE EN PHRASES
# =====================================================

def get_sentences(data: dict) -> List[str]:
    """Extract sentences from data, handling pre-tokenized and raw text."""
    # Check if already tokenized
    if isinstance(data.get("sentences"), list):
        return [s.strip() for s in data["sentences"] if str(s).strip()]
    
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Use NLTK for sentence tokenization
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback to simple splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    return [s.strip() for s in sentences if s.strip()]


# =====================================================
# STEP 5: DÉTECTION DES SECTIONS
# =====================================================

# Enhanced section patterns
SECTION_PATTERNS = {
    "definition": re.compile(
        r"^(Definition|Define|Définition)\b[:\s]*(.*)$",
        re.IGNORECASE
    ),
    "theorem": re.compile(
        r"^(Theorem|Lemma|Corollary|Proposition|Théorème)\b[:\s]*(.*)$",
        re.IGNORECASE
    ),
    "proof": re.compile(
        r"^(Proof|Démonstration)\b[:\s]*(.*)$",
        re.IGNORECASE
    ),
    "example": re.compile(
        r"^(Example|Exercise|Problem|Exemple|Exercice)\b[:\s]*(.*)$",
        re.IGNORECASE
    ),
    "remark": re.compile(
        r"^(Remark|Note|Remarque|Observation)\b[:\s]*(.*)$",
        re.IGNORECASE
    ),
    "solution": re.compile(
        r"^(Solution)\b[:\s]*(.*)$",
        re.IGNORECASE
    ),
}

# Sections to skip (non-pedagogical)
SKIP_SECTIONS = {"introduction", "about", "license", "copyright", "preface", "acknowledgment"}


def detect_section_type(sentence: str) -> Tuple[str, Optional[str]]:
    """Detect section type and extract title from a sentence."""
    for section_type, pattern in SECTION_PATTERNS.items():
        match = pattern.match(sentence.strip())
        if match:
            title = match.group(2).strip() if match.lastindex >= 2 else None
            return section_type, title
    return "section", None


def detect_sections(sentences: List[str]) -> List[Dict]:
    """Split sentences into sections with type and title."""
    sections = []
    current = {
        "type": "section",
        "title": None,
        "sentences": []
    }
    
    for sentence in sentences:
        section_type, title = detect_section_type(sentence)
        
        if section_type != "section":
            # New section detected
            if current["sentences"]:
                sections.append(current)
            current = {
                "type": section_type,
                "title": title or sentence[:100],
                "sentences": [sentence]
            }
        else:
            current["sentences"].append(sentence)
    
    # Add last section
    if current["sentences"]:
        sections.append(current)
    
    # Filter out non-pedagogical sections
    return [s for s in sections if s["type"] not in SKIP_SECTIONS]


# =====================================================
# STEP 6: CRÉATION DES CHUNKS
# =====================================================

def contains_formula(text: str) -> bool:
    """Check if text contains mathematical formulas."""
    return bool(re.search(r"[=\^∫Σπ√<>∀∃∈∉⊂⊃∪∩±×÷≤≥≠≈∞]", text))


def group_formula_sentences(sentences: List[str]) -> List[str]:
    """Group formula-containing sentences with their context."""
    grouped = []
    for sentence in sentences:
        if contains_formula(sentence) and grouped:
            grouped[-1] += " " + sentence
        else:
            grouped.append(sentence)
    return grouped


def create_chunks(
    sentences: List[str],
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 60
) -> List[str]:
    """Create chunks from sentences with overlap."""
    if not sentences:
        return []
    
    chunks = []
    buffer = []
    buffer_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence exceeds max, create chunk
        if buffer_tokens + sentence_tokens > max_tokens and buffer:
            chunk = " ".join(buffer)
            chunks.append(chunk)
            
            # Create overlap from end of buffer
            overlap = []
            overlap_count = 0
            for prev in reversed(buffer):
                prev_tokens = count_tokens(prev)
                if overlap_count + prev_tokens > overlap_tokens:
                    break
                overlap.insert(0, prev)
                overlap_count += prev_tokens
            
            buffer = overlap
            buffer_tokens = overlap_count
        
        buffer.append(sentence)
        buffer_tokens += sentence_tokens
    
    # Handle remaining buffer
    if buffer:
        chunk = " ".join(buffer)
        if count_tokens(chunk) >= min_tokens or not chunks:
            chunks.append(chunk)
        elif chunks:
            # Merge with last chunk if too small
            chunks[-1] += " " + chunk
    
    return chunks


# =====================================================
# STEP 7: AJOUT DES HEADERS CONTEXTUELS
# =====================================================

SECTION_LABELS = {
    "definition": "DEFINITION",
    "theorem": "THEOREM",
    "proof": "PROOF",
    "example": "EXAMPLE",
    "remark": "REMARK",
    "solution": "SOLUTION",
    "section": "CONTENT",
}


def add_contextual_header(
    text: str,
    course: str,
    topic: str,
    section_type: str,
    section_title: Optional[str]
) -> str:
    """Add contextual header to chunk text."""
    header_parts = []
    
    # Course and topic header
    if topic and topic != "General":
        header_parts.append(f"[{course} - {topic}]")
    else:
        header_parts.append(f"[{course}]")
    
    # Section type label
    section_label = SECTION_LABELS.get(section_type, "CONTENT")
    header_parts.append(f"[{section_label}]")
    
    # Section title if available
    if section_title:
        header_parts.append(section_title)
    
    header = "\n".join(header_parts)
    return f"{header}\n\n{text}"


# =====================================================
# STEP 8: VALIDATION DE QUALITÉ
# =====================================================

def validate_chunk(chunk: Dict) -> bool:
    """Validate chunk quality."""
    tokens = chunk.get("tokens", 0)
    text = chunk.get("text_clean", "")
    
    # Token count validation
    if tokens < CONFIG["min_tokens"]:
        return False
    if tokens > CONFIG["max_tokens"]:
        return False
    
    # Content quality validation
    if not text.strip():
        return False
    
    # Check for too much noise (very short words only)
    words = text.split()
    if len(words) < 10:
        return False
    
    # Check average word length (too short = likely noise)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 2:
        return False
    
    return True


# =====================================================
# STEP 9: DÉDUPLICATION
# =====================================================

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks based on text hash."""
    seen = set()
    unique = []
    
    for chunk in chunks:
        # Hash the clean text (lowercase, normalized whitespace)
        text_normalized = " ".join(chunk["text_clean"].lower().split())
        hash_value = hashlib.md5(text_normalized.encode()).hexdigest()
        
        if hash_value not in seen:
            seen.add(hash_value)
            unique.append(chunk)
    
    return unique


# =====================================================
# STEP 2 & 10: LECTURE DES FICHIERS & TRAITEMENT
# =====================================================

def process_file(file_path: Path) -> List[Dict]:
    """Process a single file through the chunking pipeline."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    filename = data.get("filename", file_path.stem)
    
    # Step 4: Get sentences
    sentences = get_sentences(data)
    if not sentences:
        return []
    
    # Step 3: Detect course
    full_text = " ".join(sentences)
    course = detect_course(full_text, filename)
    
    # Step 5: Detect sections
    sections = detect_sections(sentences)
    
    chunks = []
    chunk_id = 0
    
    for section in sections:
        section_type = section["type"]
        section_title = section["title"]
        section_sentences = section["sentences"]
        
        # Step 3: Detect topic for this section
        section_text = " ".join(section_sentences)
        topic = detect_topic(section_text, section_type, section_title or "")
        
        # Step 6: Group formulas and create chunks
        grouped = group_formula_sentences(section_sentences)
        section_chunks = create_chunks(
            grouped,
            min_tokens=CONFIG["target_min_tokens"],
            max_tokens=CONFIG["target_max_tokens"],
            overlap_tokens=CONFIG["overlap_tokens"]
        )
        
        for text in section_chunks:
            text_clean = text.strip()
            tokens = count_tokens(text_clean)
            
            # Step 7: Add contextual header
            text_with_header = add_contextual_header(
                text_clean,
                course,
                topic,
                section_type,
                section_title
            )
            
            chunk = {
                "source": filename,
                "course": course,
                "topic": topic,
                "section": section_type,
                "section_title": section_title or f"{section_type.title()} {chunk_id + 1}",
                "chunk_id": chunk_id,
                "text": text_with_header,
                "text_clean": text_clean,
                "tokens": count_tokens(text_with_header)
            }
            
            # Step 8: Validate
            if validate_chunk(chunk):
                chunks.append(chunk)
                chunk_id += 1
    
    return chunks


def main():
    """Main processing pipeline."""
    
    # STEP 2: LECTURE DES FICHIERS
    print("\n" + "=" * 60)
    print("STEP 2: LECTURE DES FICHIERS")
    print("=" * 60)
    
    input_files = list(CONFIG["input_dir"].glob("*.json"))
    print(f"✅ Found {len(input_files)} JSON files")
    
    # Process all files
    all_chunks = []
    course_stats = Counter()
    section_stats = Counter()
    failed_files = []
    
    print("\n" + "=" * 60)
    print("STEPS 3-8: PROCESSING FILES")
    print("=" * 60)
    
    for i, file_path in enumerate(input_files, 1):
        try:
            file_chunks = process_file(file_path)
            all_chunks.extend(file_chunks)
            
            # Collect stats
            for chunk in file_chunks:
                course_stats[chunk["course"]] += 1
                section_stats[chunk["section"]] += 1
            
            if i % 50 == 0 or i == len(input_files):
                print(f"✓ Processed {i}/{len(input_files)} files | Total chunks: {len(all_chunks)}")
                
        except Exception as e:
            failed_files.append((file_path.name, str(e)))
            if len(failed_files) <= 5:
                print(f"❌ Failed {file_path.name}: {e}")
    
    # STEP 9: DÉDUPLICATION
    print("\n" + "=" * 60)
    print("STEP 9: DÉDUPLICATION")
    print("=" * 60)
    
    original_count = len(all_chunks)
    all_chunks = deduplicate_chunks(all_chunks)
    removed = original_count - len(all_chunks)
    print(f"✅ Removed {removed} duplicate chunks")
    print(f"✅ Final count: {len(all_chunks)} chunks")
    
    # STEP 10: SAUVEGARDE ET ANALYSE
    print("\n" + "=" * 60)
    print("STEP 10: SAUVEGARDE ET ANALYSE")
    print("=" * 60)
    
    # Save chunks
    output_file = CONFIG["output_dir"] / "all_chunks_v3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to: {output_file}")
    
    # Analysis
    print("\n" + "-" * 60)
    print("ANALYSIS REPORT")
    print("-" * 60)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Files processed: {len(input_files) - len(failed_files)}")
    print(f"  Files failed: {len(failed_files)}")
    
    if all_chunks:
        tokens_list = [c["tokens"] for c in all_chunks]
        print(f"\n📊 TOKEN STATISTICS:")
        print(f"  Min tokens: {min(tokens_list)}")
        print(f"  Max tokens: {max(tokens_list)}")
        print(f"  Avg tokens: {sum(tokens_list) / len(tokens_list):.1f}")
    
    print(f"\n📊 CHUNKS BY COURSE:")
    for course, count in course_stats.most_common():
        print(f"  - {course}: {count}")
    
    print(f"\n📊 CHUNKS BY SECTION TYPE:")
    for section, count in section_stats.most_common():
        print(f"  - {section}: {count}")
    
    if failed_files:
        print(f"\n⚠️  FAILED FILES ({len(failed_files)}):")
        for name, error in failed_files[:10]:
            print(f"  - {name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    # Show sample chunk
    if all_chunks:
        print("\n" + "-" * 60)
        print("SAMPLE CHUNK (first)")
        print("-" * 60)
        sample = all_chunks[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("✅ CHUNKING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
