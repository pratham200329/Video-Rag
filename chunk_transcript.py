import json

INPUT_FILE = "audio.json"
OUTPUT_FILE = "chunks.json"

MAX_CHARS = 500

# Load Whisper output
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]

chunks = []
current_text = []
current_start = None
current_end = None
current_length = 0

for segment in segments:
    text = segment["text"].strip()

    if not text:
        continue

    if current_start is None:
        current_start = segment["start"]

    current_text.append(text)
    current_end = segment["end"]
    current_length += len(text)

    # Chunk complete
    if current_length >= MAX_CHARS:
        chunks.append({
            "text": " ".join(current_text),
            "start": current_start,
            "end": current_end
        })

        current_text = []
        current_start = None
        current_end = None
        current_length = 0

# Add remaining text
if current_text:
    chunks.append({
        "text": " ".join(current_text),
        "start": current_start,
        "end": current_end
    })

# Save chunks
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Created {len(chunks)} chunks.")