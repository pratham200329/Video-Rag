import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("video.index")

# Load chunks
with open("embedded_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


# User question
question = input("Ask a question: ")

# Convert question into embedding
question_embedding = model.encode([question])

# Convert to FAISS-compatible format
question_embedding = np.array(
    question_embedding,
    dtype="float32"
)

# Search top 3 relevant chunks
distances, indices = index.search(question_embedding, 3)

# Display results
print("\nRelevant chunks:\n")

for rank, (distance, index_id) in enumerate(
    zip(distances[0], indices[0]), 1
):
    chunk = chunks[index_id]

    print(f"--- Result {rank} ---")
    print(f"Distance: {distance:.4f}")
    print(f"Timestamp: {chunk['start']:.2f}s - {chunk['end']:.2f}s")
    print(f"Text: {chunk['text']}")
    print()