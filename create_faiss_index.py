import json
import numpy as np
import faiss

# Load embedded chunks
with open("embedded_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract embeddings
embeddings = np.array(
    [chunk["embedding"] for chunk in chunks],
    dtype="float32"
)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

# Save index
faiss.write_index(index, "video.index")

print(f"FAISS index created.")
print(f"Vectors stored: {index.ntotal}")
print(f"Vector dimension: {dimension}")