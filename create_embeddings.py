import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load chunks
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract text
texts = [chunk["text"] for chunk in chunks]

# Create embeddings for all chunks
embeddings = model.encode(texts)

# Attach embedding to each chunk
for chunk, embedding in zip(chunks, embeddings):
    chunk["embedding"] = embedding.tolist()

# Save result
with open("embedded_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Created embeddings for {len(chunks)} chunks.")
print(f"Vector dimension: {len(embeddings[0])}")