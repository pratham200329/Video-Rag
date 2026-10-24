import os
import re
import json
import time
import hashlib
import shutil
import subprocess
import logging
import traceback
import yt_dlp
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger("videorag.backend")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def sanitize_error_detail(err: Exception) -> str:
    """Return a descriptive, safe error message without leaking sensitive credentials or tokens."""
    msg = str(err)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and len(groq_key) > 5 and groq_key in msg:
        msg = msg.replace(groq_key, "[REDACTED_KEY]")
    msg = re.sub(r'gsk_[A-Za-z0-9_-]{20,}', '[REDACTED_KEY]', msg)
    msg = re.sub(r'Bearer\s+[A-Za-z0-9._-]+', 'Bearer [REDACTED]', msg)
    return msg

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from .pipeline import (
        transcribe_audio,
        create_chunks,
        create_embeddings,
        create_faiss_index,
        retrieve_chunks,
        retrieve_global_chunks,
        generate_answer
    )
except (ImportError, ValueError):
    from pipeline import (
        transcribe_audio,
        create_chunks,
        create_embeddings,
        create_faiss_index,
        retrieve_chunks,
        retrieve_global_chunks,
        generate_answer
    )


# =========================================================
# CLEAN LLM RESPONSE
# =========================================================

def clean_response(text):

    # Remove ANSI escape/control sequences
    text = re.sub(
        r'(?:\x1B|\ufffd)?\[[0-9;?]*[ -/]*[@-~]',
        '',
        text
    )

    # Remove replacement characters
    text = text.replace('\ufffd', '')

    # Remove other control characters
    text = re.sub(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',
        '',
        text
    )

    return text.strip()


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="VideoRAG AI"
)


# =========================================================
# UPLOAD DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# =========================================================
# CORS
# =========================================================

# Built-in allowed origins: local dev + known production frontends.
# To add more origins at deploy time, set the CORS_ORIGINS env var as a
# comma-separated list, e.g.:
#   CORS_ORIGINS=https://my-app.vercel.app,https://custom-domain.com
_BUILTIN_ORIGINS = [
    # Local development
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Production Vercel frontend
    "https://frontend-theta-lemon-qnvdcsvxf.vercel.app",
]

allowed_origins = list(_BUILTIN_ORIGINS)  # mutable copy

# Merge any runtime-configured origins (e.g. custom domains, PR previews)
_custom_origins_env = os.getenv("CORS_ORIGINS", "").strip()
if _custom_origins_env:
    for _o in _custom_origins_env.split(","):
        _o = _o.strip().rstrip("/")
        if _o and _o not in allowed_origins:
            allowed_origins.append(_o)
            logger.info(f"CORS: added runtime origin '{_o}' from CORS_ORIGINS env var")

logger.info(f"CORS: allowed origins = {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range"],
    max_age=600,  # preflight cache: 10 minutes
)


# =========================================================
# SERVE VIDEO FILES
# =========================================================

app.mount(
    "/uploads",
    StaticFiles(
        directory=UPLOAD_DIR
    ),
    name="uploads"
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message":
        "VideoRAG Backend is running!"
    }


# =========================================================
# VIDEO CATALOG & PERSISTENCE
# =========================================================

CATALOG_PATH = os.path.join(UPLOAD_DIR, "videos.json")


def load_catalog():
    if os.path.exists(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading catalog: {e}")
    return []


def save_catalog(catalog):
    try:
        temp_path = CATALOG_PATH + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        if os.path.exists(CATALOG_PATH):
            os.replace(temp_path, CATALOG_PATH)
        else:
            os.rename(temp_path, CATALOG_PATH)
    except Exception as e:
        print(f"Error saving catalog: {e}")


def get_file_hash(file_path):
    """Compute MD5 hash to detect duplicate video files quickly."""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def sync_existing_videos():
    """Auto-discover pre-existing videos on startup and register them in videos.json."""
    catalog = load_catalog()
    catalog_by_filename = {v.get("filename"): v for v in catalog if v.get("filename")}

    SUPPORTED_EXT = (".mp4", ".mov", ".mkv", ".webm")
    changed = False

    if not os.path.exists(UPLOAD_DIR):
        return

    for file in os.listdir(UPLOAD_DIR):
        if file.endswith(SUPPORTED_EXT):
            video_path = os.path.join(UPLOAD_DIR, file)
            base_name = os.path.splitext(file)[0]
            audio_path = os.path.join(UPLOAD_DIR, f"{base_name}.wav")
            json_path = os.path.join(UPLOAD_DIR, f"{base_name}.json")
            chunks_path = os.path.join(UPLOAD_DIR, f"{base_name}_chunks.json")
            embeddings_path = os.path.join(UPLOAD_DIR, f"{base_name}_chunks_embedded.json")
            index_path = os.path.join(UPLOAD_DIR, f"{base_name}_chunks_embedded.index")

            is_ready = os.path.exists(embeddings_path) and os.path.exists(index_path)

            if file not in catalog_by_filename:
                file_hash = get_file_hash(video_path)
                item = {
                    "id": f"vid_{base_name}",
                    "filename": file,
                    "title": file,
                    "video_path": video_path,
                    "audio_path": audio_path if os.path.exists(audio_path) else None,
                    "transcript_path": json_path if os.path.exists(json_path) else None,
                    "chunks_path": chunks_path if os.path.exists(chunks_path) else None,
                    "embeddings_path": embeddings_path if os.path.exists(embeddings_path) else None,
                    "faiss_index_path": index_path if os.path.exists(index_path) else None,
                    "file_hash": file_hash,
                    "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
                    "status": "ready" if is_ready else "indexed" if os.path.exists(chunks_path) else "uploaded",
                    "created_at": os.path.getmtime(video_path)
                }
                catalog.append(item)
                catalog_by_filename[file] = item
                changed = True
            else:
                item = catalog_by_filename[file]
                if is_ready and item.get("status") != "ready":
                    item["status"] = "ready"
                    item["embeddings_path"] = embeddings_path
                    item["faiss_index_path"] = index_path
                    changed = True

    if changed or not os.path.exists(CATALOG_PATH):
        save_catalog(catalog)


@app.on_event("startup")
def on_startup():
    sync_existing_videos()


# =========================================================
# GET ALL VIDEOS (LIBRARY CATALOG)
# =========================================================

@app.get("/videos")
def get_videos():
    sync_existing_videos()
    return {
        "videos": load_catalog()
    }


# =========================================================
# UPLOAD + COMPLETE PROCESSING
# =========================================================

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):
    try:
        base_name = os.path.splitext(file.filename)[0]
        video_path = os.path.join(UPLOAD_DIR, file.filename)
        audio_path = os.path.join(UPLOAD_DIR, base_name + ".wav")
        json_path = os.path.join(UPLOAD_DIR, base_name + ".json")
        chunks_path = os.path.join(UPLOAD_DIR, base_name + "_chunks.json")
        embeddings_path = os.path.join(UPLOAD_DIR, base_name + "_chunks_embedded.json")
        index_path = os.path.join(UPLOAD_DIR, base_name + "_chunks_embedded.index")

        logger.info(f"Initiating upload for video: {file.filename}")

        # 1. Save video
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_hash = get_file_hash(video_path)

        # Check if duplicate already indexed
        catalog = load_catalog()
        existing = next(
            (v for v in catalog if v.get("filename") == file.filename or (file_hash and v.get("file_hash") == file_hash)),
            None
        )
        if (
            existing
            and existing.get("status") == "ready"
            and existing.get("embeddings_path")
            and os.path.exists(existing.get("embeddings_path", ""))
            and existing.get("faiss_index_path")
            and os.path.exists(existing.get("faiss_index_path", ""))
        ):
            logger.info(f"Video {file.filename} already indexed. Returning cached metadata.")
            print(f"Video {file.filename} already indexed. Returning cached metadata.")
            return {
                **existing,
                "message": "Video RAG indexing completed successfully! (cached)"
            }

        # 2. Extract audio with FFmpeg
        if os.path.exists(audio_path):
            logger.info("Audio already exists. Skipping FFmpeg.")
            print("Audio already exists. Skipping FFmpeg.")
        else:
            logger.info(f"Extracting audio from {video_path}...")
            print("Extracting audio...")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-i",
                        video_path,
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-y",
                        audio_path
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError as ff_err:
                logger.error(f"FFmpeg audio extraction failed (exit {ff_err.returncode}): {ff_err.stderr}")
                raise RuntimeError(f"FFmpeg audio extraction failed: {ff_err.stderr or ff_err}") from ff_err

        # 3. Whisper speech-to-text
        logger.info(f"Starting Whisper transcription for {audio_path}...")
        json_path = transcribe_audio(audio_path)

        # 4. Chunking
        logger.info(f"Generating chunks for {json_path}...")
        chunks_path = create_chunks(json_path)

        # 5. Embeddings
        logger.info(f"Generating embeddings for {chunks_path}...")
        embeddings_path = create_embeddings(chunks_path)

        # 6. FAISS index
        logger.info(f"Creating FAISS vector index for {embeddings_path}...")
        index_path = create_faiss_index(embeddings_path)

        # 7. Register in persistent library catalog
        video_entry = {
            "id": f"vid_{base_name}",
            "filename": file.filename,
            "title": file.filename,
            "video_path": video_path,
            "audio_path": audio_path,
            "transcript_path": json_path,
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "faiss_index_path": index_path,
            "file_hash": file_hash,
            "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
            "status": "ready",
            "created_at": time.time()
        }

        catalog = [v for v in catalog if v.get("filename") != file.filename and v.get("id") != video_entry["id"]]
        catalog.append(video_entry)
        save_catalog(catalog)

        logger.info(f"Video {file.filename} processing and cataloging complete.")
        return {
            **video_entry,
            "message": "Video RAG indexing completed successfully!"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error during video upload processing for {getattr(file, 'filename', 'unknown')}: {e}")
        safe_detail = sanitize_error_detail(e)
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {safe_detail}"
        )


# =========================================================
# PROCESS VIDEO LINK (DOWNLOAD + INDEX)
# =========================================================

class LinkRequest(BaseModel):
    url: str


@app.post("/process-link")
async def process_video_link(
    req: LinkRequest
):
    url = req.url.strip()
    if not url:
        raise HTTPException(
            status_code=400,
            detail="Video URL cannot be empty."
        )

    logger.info(f"Downloading video from link: {url}")
    print(f"Downloading video from link: {url}")

    ydl_opts = {
        'outtmpl': os.path.join(UPLOAD_DIR, '%(id)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'noplaylist': True,
        'quiet': False,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "web_video")
            filename = f"{video_id}.mp4"
            video_path = os.path.join(UPLOAD_DIR, filename)

            if not os.path.exists(video_path):
                prep = ydl.prepare_filename(info)
                base, _ = os.path.splitext(prep)
                if os.path.exists(base + ".mp4"):
                    video_path = base + ".mp4"
                    filename = os.path.basename(video_path)
                elif os.path.exists(prep):
                    video_path = prep
                    filename = os.path.basename(video_path)

    except Exception as e:
        logger.exception(f"Error downloading video from link {url}: {e}")
        print(f"Error downloading video: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download video: {sanitize_error_detail(e)}"
        )

    try:
        base_name = os.path.splitext(filename)[0]
        audio_path = os.path.join(UPLOAD_DIR, base_name + ".wav")
        json_path = os.path.join(UPLOAD_DIR, base_name + ".json")
        chunks_path = os.path.join(UPLOAD_DIR, base_name + "_chunks.json")
        embeddings_path = os.path.join(UPLOAD_DIR, base_name + "_chunks_embedded.json")
        index_path = os.path.join(UPLOAD_DIR, base_name + "_chunks_embedded.index")

        file_hash = get_file_hash(video_path)

        # Check if duplicate already indexed
        catalog = load_catalog()
        existing = next(
            (v for v in catalog if v.get("filename") == filename or (file_hash and v.get("file_hash") == file_hash)),
            None
        )
        if (
            existing
            and existing.get("status") == "ready"
            and existing.get("embeddings_path")
            and os.path.exists(existing.get("embeddings_path", ""))
            and existing.get("faiss_index_path")
            and os.path.exists(existing.get("faiss_index_path", ""))
        ):
            logger.info(f"Link video {filename} already indexed. Returning cached metadata.")
            print(f"Link video {filename} already indexed. Returning cached metadata.")
            return {
                **existing,
                "message": "Video link downloaded and indexed successfully! (cached)"
            }

        # 1. FFmpeg
        if not os.path.exists(audio_path):
            logger.info(f"Extracting audio with FFmpeg to {audio_path}...")
            print("Extracting audio with FFmpeg...")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-i",
                        video_path,
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-y",
                        audio_path
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError as ff_err:
                logger.error(f"FFmpeg audio extraction failed (exit {ff_err.returncode}): {ff_err.stderr}")
                raise RuntimeError(f"FFmpeg audio extraction failed: {ff_err.stderr or ff_err}") from ff_err

        # 2. Whisper
        logger.info(f"Starting Whisper transcription for {audio_path}...")
        json_path = transcribe_audio(audio_path)

        # 3. Chunking
        logger.info(f"Generating chunks for {json_path}...")
        chunks_path = create_chunks(json_path)

        # 4. Embeddings
        logger.info(f"Generating embeddings for {chunks_path}...")
        embeddings_path = create_embeddings(chunks_path)

        # 5. FAISS
        logger.info(f"Building FAISS vector index for {embeddings_path}...")
        index_path = create_faiss_index(embeddings_path)

        # 6. Register in persistent catalog
        video_entry = {
            "id": f"vid_{base_name}",
            "filename": filename,
            "title": filename,
            "video_path": video_path,
            "audio_path": audio_path,
            "transcript_path": json_path,
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "faiss_index_path": index_path,
            "file_hash": file_hash,
            "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
            "status": "ready",
            "created_at": time.time()
        }

        catalog = [v for v in catalog if v.get("filename") != filename and v.get("id") != video_entry["id"]]
        catalog.append(video_entry)
        save_catalog(catalog)

        logger.info(f"Link video {filename} processing complete.")
        return {
            **video_entry,
            "message": "Video link downloaded and indexed successfully!"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error during link video processing for {filename}: {e}")
        safe_detail = sanitize_error_detail(e)
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {safe_detail}"
        )


# =========================================================
# ASK & CHAT (MULTI-VIDEO CROSS-RETRIEVAL & MULTI-TURN RAG)
# =========================================================

class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    video_id: Optional[str] = "all"
    history: Optional[List[ChatTurn]] = None


def execute_qa(
    question: str,
    video_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None
):
    question = question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    catalog = load_catalog()
    ready_videos = [
        v for v in catalog
        if v.get("status") == "ready"
        and v.get("embeddings_path")
        and os.path.exists(v["embeddings_path"])
        and v.get("faiss_index_path")
        and os.path.exists(v["faiss_index_path"])
    ]

    # Target specific video if requested
    if video_id and video_id != "all":
        filtered = [
            v for v in ready_videos
            if v.get("id") == video_id or v.get("filename") == video_id
        ]
        if filtered:
            ready_videos = filtered
        else:
            # Check if video exists in catalog but is not ready/indexed yet
            matching = [
                v for v in catalog
                if v.get("id") == video_id or v.get("filename") == video_id
            ]
            if matching:
                vid_title = matching[0].get("title") or matching[0].get("filename") or video_id
                vid_status = matching[0].get("status", "unprocessed")
                return {
                    "question": question,
                    "answer": f"The selected video '{vid_title}' has status '{vid_status}' and has not been indexed yet. Please wait until processing is complete or index it before querying.",
                    "sources": [],
                    "ranked_videos": []
                }
            else:
                return {
                    "question": question,
                    "answer": f"The selected video '{video_id}' could not be found in the catalog.",
                    "sources": [],
                    "ranked_videos": []
                }

    results = []
    if ready_videos:
        # Cross-video retrieval across all ready video indexes
        results = retrieve_global_chunks(
            question,
            ready_videos,
            top_k_per_video=3,
            global_top_k=5
        )
    else:
        # Backward-compatibility fallback
        embeddings_path = os.path.join(
            UPLOAD_DIR,
            "lecture_chunks_embedded.json"
        )
        index_path = os.path.join(
            UPLOAD_DIR,
            "lecture_chunks_embedded.index"
        )

        if os.path.exists(embeddings_path) and os.path.exists(index_path):
            results = retrieve_chunks(
                question,
                embeddings_path,
                index_path,
                top_k=3
            )
            for r in results:
                r["filename"] = "lecture.mp4"
                r["video_id"] = "vid_lecture"

    if not results:
        return {
            "question": question,
            "answer": "I couldn't find enough information about this in any of the indexed videos.",
            "sources": [],
            "ranked_videos": []
        }

    # Generate answer with Groq API (passing conversation history if available)
    answer = generate_answer(
        question,
        results,
        history=history
    )

    answer = clean_response(
        answer
    )

    # Compute video relevance ranking
    # Smaller L2 distance = closer semantic match = higher rank
    video_groups = {}
    for r in results:
        fname = r.get("filename", "Unknown")
        vid = r.get("video_id", fname)
        dist = r.get("distance", 999.0)

        if fname not in video_groups:
            video_groups[fname] = {
                "video_id": vid,
                "filename": fname,
                "min_distance": dist,
                "chunk_count": 1
            }
        else:
            video_groups[fname]["chunk_count"] += 1
            if dist < video_groups[fname]["min_distance"]:
                video_groups[fname]["min_distance"] = dist

    # Sort ascending: lowest L2 distance ranks #1
    ranked_videos = sorted(video_groups.values(), key=lambda x: x["min_distance"])

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "video_id": result.get("video_id"),
                "filename": result.get("filename"),
                "start": result["start"],
                "end": result["end"],
                "distance": result["distance"],
                "text": result.get("text", "")
            }
            for result in results
        ],
        "ranked_videos": ranked_videos
    }


@app.get("/ask")
def ask_video(
    question: str,
    video_id: str = None
):
    return execute_qa(question=question, video_id=video_id)


@app.post("/chat")
def chat_video(
    req: ChatRequest
):
    history_dicts = None
    if req.history:
        history_dicts = [{"role": h.role, "content": h.content} for h in req.history]
    return execute_qa(question=req.question, video_id=req.video_id, history=history_dicts)