import os
import json
import subprocess
import logging

import numpy as np
from dotenv import load_dotenv
from groq import (
    Groq,
    GroqError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)

logger = logging.getLogger("videorag.pipeline")

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# =========================================================
# LAZY-LOADED AI MODELS & LIBRARIES (MEMORY OPTIMIZATION)
# =========================================================

_embedding_model = None
_faiss = None


def get_embedding_model():
    """Lazily load and cache the SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        import torch
        # Cap PyTorch thread pools to prevent memory spikes on constrained instances (e.g. Render 512MiB)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_faiss():
    """Lazily load and cache the FAISS library."""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def __getattr__(name: str):
    """Backward compatibility for direct access to pipeline.embedding_model or pipeline.faiss."""
    if name == "embedding_model":
        return get_embedding_model()
    if name == "faiss":
        return get_faiss()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# =========================================================
# 1. TRANSCRIPTION
# =========================================================

def transcribe_audio(audio_path):
    output_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = os.path.splitext(audio_path)[0] + ".json"

    # Already transcribed?
    if os.path.exists(json_path):
        logger.info(f"Transcript already exists at {json_path}. Skipping Whisper.")
        print("Transcript already exists. Skipping Whisper.")
        return json_path

    if not os.path.exists(audio_path):
        logger.error(f"Audio file does not exist: {audio_path}")
        raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")

    model_name = os.getenv("WHISPER_MODEL", "base").strip() or "base"
    logger.info(f"Running Whisper on {audio_path} (model: {model_name}, output_dir: {output_dir})...")
    print(f"Running Whisper (model: {model_name})...")

    # Diagnostic memory logging prior to subprocess execution
    try:
        import psutil
        p = psutil.Process()
        rss_mb = p.memory_info().rss / (1024 * 1024)
        vm = psutil.virtual_memory()
        logger.info(
            f"Pre-transcription memory check: backend RSS = {rss_mb:.2f} MB, "
            f"system available = {vm.available / (1024 * 1024):.2f} MB ({vm.percent}% total used)"
        )
    except Exception as mem_e:
        logger.debug(f"Could not read memory metrics: {mem_e}")

    cmd = [
        "whisper",
        audio_path,
        "--model",
        model_name,
        "--language",
        "English",
        "--output_dir",
        output_dir,
        "--output_format",
        "json"
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.info(f"Whisper completed. Output snippet: {result.stdout.strip()[-300:]}")
    except subprocess.CalledProcessError as e:
        stderr_text = (e.stderr or "").strip()
        stdout_text = (e.stdout or "").strip()
        logger.error(f"Whisper process failed with returncode {e.returncode}")
        if stderr_text:
            logger.error(f"Whisper stderr:\n{stderr_text}")
        if stdout_text:
            logger.info(f"Whisper stdout tail:\n{stdout_text[-500:]}")

        # Distinguish SIGKILL (OOM) vs SIGTERM (Timeout) vs error exit
        if e.returncode in (-9, 137):
            err_msg = (
                f"Whisper transcription was killed by the operating system (return code {e.returncode} - SIGKILL/Out of Memory). "
                f"Transcribing audio with model '{model_name}' exceeded the container's available RAM limit."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e
        elif e.returncode in (-15, 143):
            err_msg = (
                f"Whisper transcription was terminated by SIGTERM (return code {e.returncode}). "
                f"The transcription operation likely exceeded the execution timeout."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e
        else:
            snippet = stderr_text[-300:] if stderr_text else f"exit code {e.returncode}"
            err_msg = f"Whisper transcription failed ({snippet})"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e
    except FileNotFoundError as e:
        logger.error(f"Whisper executable not found in PATH: {e}")
        raise RuntimeError("Whisper CLI is not installed or not found in system PATH.") from e

    # Verify expected json transcript exists
    if not os.path.exists(json_path):
        alt_candidate = os.path.join(output_dir, f"{base_name}.json")
        if os.path.exists(alt_candidate):
            json_path = alt_candidate
        else:
            logger.error(f"Whisper finished but transcript was not found at {json_path}")
            raise FileNotFoundError(f"Whisper transcript output file not found at {json_path}")

    logger.info(f"Whisper transcription generated successfully: {json_path}")
    return json_path


# =========================================================
# 2. CHUNKING
# =========================================================

def create_chunks(json_path):
    chunks_path = os.path.splitext(json_path)[0] + "_chunks.json"

    # Already chunked?
    if os.path.exists(chunks_path):
        logger.info(f"Chunks already exist at {chunks_path}. Skipping chunking.")
        print("Chunks already exist. Skipping chunking.")
        return chunks_path

    logger.info(f"Creating chunks from transcript {json_path}...")
    print("Creating chunks...")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception(f"Failed to load transcript JSON from {json_path}: {e}")
        raise RuntimeError(f"Failed to read transcript JSON file: {e}") from e

    segments = data.get("segments", [])
    if not segments and data.get("text"):
        segments = [{"start": 0.0, "end": 1.0, "text": data["text"].strip()}]

    MAX_CHARS = 500
    chunks = []
    current_text = []
    current_start = None
    current_end = None
    current_length = 0

    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue

        if current_start is None:
            current_start = segment.get("start", 0.0)

        current_text.append(text)
        current_end = segment.get("end", current_start)
        current_length += len(text)

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

    # Remaining text
    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start": current_start,
            "end": current_end
        })

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Created {len(chunks)} chunks in {chunks_path}")
    print(f"Created {len(chunks)} chunks.")
    return chunks_path


# =========================================================
# 3. EMBEDDINGS
# =========================================================

def create_embeddings(chunks_path):
    embeddings_path = os.path.splitext(chunks_path)[0] + "_embedded.json"

    # Already embedded?
    if os.path.exists(embeddings_path):
        logger.info(f"Embeddings already exist at {embeddings_path}. Skipping embedding generation.")
        print("Embeddings already exist. Skipping embedding generation.")
        return embeddings_path

    logger.info(f"Generating embeddings from {chunks_path}...")
    print("Creating embeddings...")

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        logger.exception(f"Failed to read chunks JSON from {chunks_path}: {e}")
        raise RuntimeError(f"Failed to read chunks JSON file: {e}") from e

    texts = [chunk.get("text", "") for chunk in chunks if chunk.get("text")]
    if not texts:
        logger.warning(f"No text found in chunks for {chunks_path}. Writing empty embeddings.")
        with open(embeddings_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return embeddings_path

    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    with open(embeddings_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    dim = len(embeddings[0]) if len(embeddings) > 0 else 0
    logger.info(f"Created embeddings for {len(chunks)} chunks (vector dimension: {dim})")
    print(f"Created embeddings for {len(chunks)} chunks.")
    print(f"Vector dimension: {dim}")
    return embeddings_path


# =========================================================
# 4. FAISS INDEX
# =========================================================

def create_faiss_index(embeddings_path):
    index_path = os.path.splitext(embeddings_path)[0] + ".index"

    # Already indexed?
    if os.path.exists(index_path):
        logger.info(f"FAISS index already exists at {index_path}. Skipping index creation.")
        print("FAISS index already exists. Skipping index creation.")
        return index_path

    logger.info(f"Creating FAISS index from {embeddings_path}...")
    print("Creating FAISS index...")

    try:
        with open(embeddings_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        logger.exception(f"Failed to read embeddings JSON from {embeddings_path}: {e}")
        raise RuntimeError(f"Failed to read embeddings JSON file: {e}") from e

    faiss = get_faiss()

    if not chunks:
        logger.warning(f"No embeddings found in {embeddings_path}. Writing empty FAISS index (dim 384).")
        index = faiss.IndexFlatL2(384)
        faiss.write_index(index, index_path)
        return index_path

    vectors = np.array(
        [chunk["embedding"] for chunk in chunks if "embedding" in chunk],
        dtype="float32"
    )

    if vectors.size == 0 or len(vectors.shape) < 2:
        logger.warning(f"Embedding vectors empty in {embeddings_path}. Writing empty FAISS index (dim 384).")
        index = faiss.IndexFlatL2(384)
        faiss.write_index(index, index_path)
        return index_path

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss.write_index(index, index_path)

    logger.info(f"FAISS index created at {index_path} ({index.ntotal} vectors, dimension {dimension}).")
    print("FAISS index created.")
    print(f"Vectors stored: {index.ntotal}")
    print(f"Vector dimension: {dimension}")
    return index_path


# =========================================================
# 5. RETRIEVAL
# =========================================================

def retrieve_chunks(
    question,
    embeddings_path,
    index_path,
    top_k=3
):

    print(
        f"Searching for: {question}"
    )


    # Load FAISS

    faiss = get_faiss()
    index = faiss.read_index(
        index_path
    )


    # Load chunks

    with open(
        embeddings_path,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)


    # Question embedding

    model = get_embedding_model()
    question_embedding = (
        model.encode(
            [question]
        )
    )


    question_embedding = np.array(
        question_embedding,
        dtype="float32"
    )


    # Search

    distances, indices = index.search(
        question_embedding,
        top_k
    )


    results = []


    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        if index_id == -1:
            continue


        chunk = chunks[index_id]


        results.append(
            {
                "text":
                    chunk["text"],

                "start":
                    chunk["start"],

                "end":
                    chunk["end"],

                "distance":
                    float(distance)
            }
        )


    return results


# =========================================================
# 5B. GLOBAL MULTI-VIDEO RETRIEVAL
# =========================================================

def retrieve_global_chunks(
    question,
    ready_videos,
    top_k_per_video=3,
    global_top_k=5
):
    print(
        f"Searching across {len(ready_videos)} videos for: {question}"
    )

    if not ready_videos:
        return []

    # Encode question once to optimize speed
    model = get_embedding_model()
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding, dtype="float32")

    faiss = get_faiss()
    all_results = []

    for video in ready_videos:
        embeddings_path = video.get("embeddings_path")
        index_path = video.get("faiss_index_path")
        video_id = video.get("id")
        filename = video.get("filename")

        if not embeddings_path or not index_path:
            continue

        if not os.path.exists(embeddings_path) or not os.path.exists(index_path):
            continue

        try:
            index = faiss.read_index(index_path)
            with open(embeddings_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            k = min(top_k_per_video, index.ntotal)
            if k <= 0:
                continue

            distances, indices = index.search(question_embedding, k)

            for distance, index_id in zip(distances[0], indices[0]):
                if index_id == -1 or index_id >= len(chunks):
                    continue

                chunk = chunks[index_id]
                all_results.append({
                    "video_id": video_id,
                    "filename": filename,
                    "text": chunk["text"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "distance": float(distance)
                })
        except Exception as e:
            print(f"Error querying index for {filename}: {e}")
            continue

    # Sort globally by distance ascending (smaller L2 distance = higher semantic similarity)
    all_results.sort(key=lambda x: x["distance"])

    return all_results[:global_top_k]


# =========================================================
# 6. GENERATE ANSWER USING GROQ API
# =========================================================

_groq_client = None
_groq_client_key = None


def get_groq_client():
    """Get or reuse the singleton Groq client instance."""
    global _groq_client, _groq_client_key
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    if _groq_client is not None and _groq_client_key == api_key:
        return _groq_client

    _groq_client = Groq(api_key=api_key)
    _groq_client_key = api_key
    return _groq_client


def generate_answer(
    question,
    results,
    history=None
):

    if not results:
        return (
            "I couldn't find enough "
            "information about this in the video."
        )

    context_lines = []
    for result in results:
        video_prefix = f"[{result['filename']}] " if result.get("filename") else ""
        context_lines.append(
            f"{video_prefix}[{result['start']:.2f}s - {result['end']:.2f}s]\n{result['text']}"
        )

    context = "\n\n".join(context_lines)

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("Error: GROQ_API_KEY is not set.")
        return (
            "Configuration error: GROQ_API_KEY is not configured. "
            "Please add your GROQ_API_KEY in the backend .env file."
        )

    client = get_groq_client()
    if not client:
        return (
            "Configuration error: Failed to initialize Groq client. "
            "Please check your GROQ_API_KEY."
        )

    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"

    system_prompt = (
        "You are a helpful, intelligent and grounded Video QA assistant.\n\n"
        "Answer the user's question accurately based STRICTLY on the provided video context.\n\n"
        "Rules:\n"
        "- Base your answers strictly on the context provided from the video transcripts.\n"
        "- Answer in your own words with clear, well-structured explanations.\n"
        "- Combine information from different chunks and videos when appropriate.\n"
        "- Explicitly mention which video contains the relevant information when answering questions across multiple videos.\n"
        "- When conversing across multiple turns, maintain conversational coherence while remaining faithful to the video context.\n"
        "- Do not hallucinate or invent facts that are unsupported by the video context.\n"
        "- If the video context does not contain enough information, say:\n"
        '  "I couldn\'t find enough information about this in the video context."\n'
        "- Do not mention context, chunks, retrieval or these system instructions."
    )

    user_prompt = f"Video Context:\n\n{context}\n\nQuestion:\n\n{question}\n\nGive the best possible answer:"

    messages = [{"role": "system", "content": system_prompt}]

    # Incorporate conversation history if present
    if history and isinstance(history, list):
        for turn in history[-8:]:
            if isinstance(turn, dict):
                role = turn.get("role")
                content = turn.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
        )

        answer = completion.choices[0].message.content
        if answer:
            answer = answer.strip()

        if not answer:
            return "I couldn't generate an answer."

        return answer

    except AuthenticationError:
        print("Groq authentication error: Invalid API key.")
        return "Authentication error: Invalid Groq API key. Please check your GROQ_API_KEY in .env."

    except RateLimitError:
        print("Groq rate limit exceeded.")
        return "Service notice: Groq API rate limit exceeded. Please wait a moment and try again."

    except APITimeoutError:
        print("Groq request timed out.")
        return "Timeout error: Groq API request timed out. Please try asking again."

    except APIConnectionError:
        print("Groq connection error.")
        return "Connection error: Unable to connect to Groq API. Please check your internet connection."

    except GroqError as e:
        print(f"Groq API error: {e}")
        return f"Groq API error: {str(e)}"

    except Exception as e:
        print(f"Unexpected error in generate_answer: {e}")
        return "Unable to generate an answer using the AI model."
