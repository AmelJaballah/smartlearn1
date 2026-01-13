import json
from pathlib import Path

# Path to the folder containing the JSON files
json_folder = Path(r"c:\Users\q\Desktop\project\ExerciceGen\json_output")

# Output file for the combined data
output_file = json_folder / "combined_exercises.json"

# List to hold all combined data
combined_data = []

# Iterate through all JSON files in the folder
for json_file in json_folder.glob("*.json"):
    if json_file.name == "combined_exercises.json":
        continue  # Skip the output file if it exists
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Try to load as a single JSON object first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                combined_data.append(data)
        except json.JSONDecodeError:
            # If that fails, assume it's JSON Lines (one JSON per line)
            lines = content.splitlines()
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        combined_data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {json_file.name}: {e}")
                        continue
        
        print(f"Processed {json_file.name}")
    except Exception as e:
        print(f"Error processing {json_file.name}: {e}")

# Write the combined data to the output file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

print(f"Combined {len(combined_data)} items into {output_file}")