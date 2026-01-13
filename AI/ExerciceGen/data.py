import json
from pathlib import Path
from collections import Counter, defaultdict

# CONFIG
DATASET_PATH = Path("json_output/combined_exercises.jsonl")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

# READ & COUNT SUBJECTS & COLLECT EXAMPLES
counter = Counter()
examples = defaultdict(list)
total = 0

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        subject = entry.get("subject")
        if subject:
            subject = subject.strip()
            counter[subject] += 1
            total += 1
            if len(examples[subject]) < 2:
                examples[subject].append(entry)

# DISPLAY RESULTS
print(f"\n Total records with subject: {total}\n")
print(" SUBJECTS FOUND:\n")

for subject, count in counter.most_common():
    print(f"{subject} : {count}")
    print("Examples:")
    for i, example in enumerate(examples[subject], 1):
        print(f"  Example {i}: {example}")
    print()

# SAVE TO JSON
output = {
    "total_records_with_subject": total,
    "subjects": {}
}

for subject, count in counter.most_common():
    output["subjects"][subject] = {
        "count": count,
        "examples": examples[subject]
    }

with open("subjects_examples.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("Results saved to subjects_examples.json")
