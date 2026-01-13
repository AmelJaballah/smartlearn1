from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from sympy import sympify, simplify
from sympy.core.sympify import SympifyError

# CONFIG (MUST MATCH INDEXATION)
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_exercises"
COLLECTION_NAME = "exercise_bank"

# Model Configuration
import os
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DIFFICULTY_MAP = {
    "easy": [1, 2, 3],
    "medium": [4, 5, 6],
    "hard": [7, 8],
}

# DEVICE + EMBEDDINGS
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

embedder = SentenceTransformer(EMBEDDING_MODEL, device=device, cache_folder=MODEL_PATH)

# CHROMA (PERSISTENT)
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(COLLECTION_NAME)

# QUERY EMBEDDING (CACHE)
@lru_cache(maxsize=256)
def embed_query(text: str):
    return embedder.encode([text], normalize_embeddings=True)[0]

# RETRIEVER (ROBUST, NO FILTERING BUGS)
def retrieve_exercises(query: str, difficulty_label: str, k: int = 5) -> List[Dict]:
    q_emb = embed_query(query)

    res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=30,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for d, m, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        retrieved.append({
            "question": d,
            "answer": m.get("answer", ""),
            "difficulty": m.get("difficulty"),
            "distance": dist,
        })

    retrieved.sort(key=lambda x: x["distance"])

    diff_vals = DIFFICULTY_MAP.get(difficulty_label.lower())
    if diff_vals:
        filtered = [r for r in retrieved if r["difficulty"] and int(r["difficulty"]) in diff_vals]
        if filtered:
            retrieved = filtered

    return retrieved[:k]

# SYMPY VALIDATOR
def verify_student_answer(expected: str, student: str) -> bool:
    try:
        return simplify(sympify(expected) - sympify(student)) == 0
    except (SympifyError, TypeError):
        return False

# LLM (STRICT SYSTEM PROMPT)
llm = ChatOllama(
    model="llama3.1",
    temperature=0.0,   # 🔥 critical for format stability
)

SYSTEM_PROMPT = """
You are a STRICT JSON GENERATOR and a PROFESSIONAL MATH TEACHER.

Rules:
- You MUST output ONLY valid JSON.
- NO text outside JSON.
- NO markdown.
- NO explanations outside JSON.
- The "solution" field MUST contain a step-by-step explanation.
- The "final_answer" field MUST contain the final numeric or symbolic answer.
- The "hint" field MUST contain a brief hint to help students.
- Do NOT leave any field empty.
- If you cannot comply, output {}.

JSON format:
{
  "question": "...",
  "solution": "Step 1: ... Step 2: ... Step 3: ...",
  "final_answer": "...",
  "hint": "..."
}
"""


@dataclass
class GeneratedExercise:
    question: str
    solution: str
    final_answer: str
    hint: str

# JSON EXTRACTION (VERY IMPORTANT)
def extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group(0))

# GENERATION (SAFE & RETRY)
def generate_exercise(subject: str, difficulty: str) -> GeneratedExercise:
    examples = retrieve_exercises(
        query=f"{subject} exercise",
        difficulty_label=difficulty,
    )

    context = "\n".join(f"- {e['question']}" for e in examples)

    prompt = f"""
{SYSTEM_PROMPT}

Context examples (for style and difficulty only):
{context}

Generate ONE new {subject} exercise.

Difficulty: {difficulty}

Pedagogical requirements:
- Explain EACH step clearly in the solution
- Use correct mathematical reasoning
- Ensure the final answer is consistent with the solution

Return ONLY the JSON.
"""

    for attempt in range(2):  # 🔁 retry once
        raw = llm.invoke(prompt).content.strip()
        try:
            data = extract_json(raw)
            return GeneratedExercise(
                question=data["question"].strip(),
                solution=data["solution"].strip(),
                final_answer=data["final_answer"].strip(),
                hint=data["hint"].strip(),
            )
        except Exception:
            if attempt == 1:
                raise RuntimeError(
                    "LLM did not return valid JSON.\n\nOutput was:\n" + raw
                )

# CLI
def main():
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--difficulty", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--student-answer")
    parser.add_argument("--show-metrics", action="store_true", help="Show detailed metrics")

    args = parser.parse_args()
    
    start_time = time.time()
    
    # Retrieve examples for metrics
    examples = retrieve_exercises(
        query=f"{args.subject} exercise",
        difficulty_label=args.difficulty,
    )
    
    exercise = generate_exercise(args.subject, args.difficulty)

    print("\n📝 QUESTION:\n", exercise.question)
    print("\n� HINT:\n", exercise.hint)
    print("\n�📘 SOLUTION:\n", exercise.solution)
    print("\n✅ FINAL ANSWER:\n", exercise.final_answer)

    if args.student_answer:
        ok = verify_student_answer(exercise.final_answer, args.student_answer)
        print("\nVERIFICATION:", "✅ Correct" if ok else "❌ Incorrect")
    
    # Calculate metrics if requested
    if args.show_metrics:
        elapsed = time.time() - start_time
        
        # Context overlap: how much of retrieved examples appear in generated exercise
        context_text = " ".join([e['question'].lower() for e in examples])
        exercise_text = f"{exercise.question} {exercise.solution}".lower()
        
        context_words = set(context_text.split())
        exercise_words = set(exercise_text.split())
        common_words = context_words.intersection(exercise_words)
        
        overlap_ratio = len(common_words) / len(exercise_words) if exercise_words else 0
        
    
        # RAG grounding: is answer based on retrieved examples?
        grounded = overlap_ratio > 0.1  
        print(f"\n{'='*60}")
        print("📊 METRICS")
        print(f"{'='*60}")
        print(f"⏱  Time:             {elapsed:.1f}s")
        print(f"📚 Examples Used:    {len(examples)}")
        print(f"🔗 Context Overlap:  {overlap_ratio:.1%}")
        print(f"✅ RAG Grounded:     {'Yes' if grounded else 'No'}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()