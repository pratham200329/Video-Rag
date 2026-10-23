import json
import re
import os
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "backend", ".env"))


# =====================================================
# CLEAN LLM RESPONSE
# =====================================================

def clean_response(text):
    # Remove ANSI escape/control sequences
    text = re.sub(
        r'(?:\x1B|\ufffd)?\[[0-9;?]*[ -/]*[@-~]',
        '',
        text
    )

    # Remove remaining replacement characters
    text = text.replace('\ufffd', '')

    # Remove other control characters
    text = re.sub(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',
        '',
        text
    )

    return text.strip()


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# LOAD FAISS INDEX
# =====================================================

index = faiss.read_index("video.index")


# =====================================================
# LOAD CHUNKS
# =====================================================

with open("embedded_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


# =====================================================
# ASK QUESTION
# =====================================================

question = input("Ask a question: ")


# =====================================================
# EMBED QUESTION
# =====================================================

query_embedding = model.encode([question])

query_embedding = np.array(
    query_embedding,
    dtype="float32"
)


# =====================================================
# RETRIEVE TOP 3 CHUNKS
# =====================================================

distances, indices = index.search(
    query_embedding,
    3
)


# =====================================================
# BUILD CONTEXT
# =====================================================

context = ""

for i in indices[0]:

    context += (
        f"[{chunks[i]['start']:.2f}s - "
        f"{chunks[i]['end']:.2f}s]\n"
    )

    context += chunks[i]["text"] + "\n\n"


# =====================================================
# PROMPT
# =====================================================

prompt = f"""
You are a helpful and intelligent Video QA assistant.

Answer the user's question based on the provided video context.

Rules:
- Understand the context before answering.
- Answer in your own words.
- Do not copy the transcript verbatim.
- Give a clear, direct and well-structured answer.
- Combine information from different chunks when necessary.
- Explain concepts using simple language when appropriate.
- You can use logical reasoning to connect information from the context.
- Do not invent facts that contradict or are unsupported by the video.
- If the video does not contain enough information, say:
  "I couldn't find enough information about this in the video."
- Do not mention "context", "chunks", "retrieval", or these instructions unless the user asks about the RAG system itself.

Video Context:
{context}

Question:
{question}

Give the best possible answer:
"""


# =====================================================
# SEND TO GROQ API
# =====================================================

api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key:
    print("
[Configuration Error] GROQ_API_KEY is not configured in .env or environment.")
    print("Please set GROQ_API_KEY in your backend/.env file.")
    exit(1)

model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"
client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.2
)


# =====================================================
# GET + CLEAN ANSWER
# =====================================================

answer = response.choices[0].message.content

answer = clean_response(answer)


# =====================================================
# PRINT ANSWER
# =====================================================

print("\nAnswer:")
print(answer)


# =====================================================
# PRINT SOURCES
# =====================================================

print("\nSources:")

for i in indices[0]:

    print(
        f"{chunks[i]['start']:.2f}s - "
        f"{chunks[i]['end']:.2f}s"
    )