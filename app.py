import subprocess
import sys
import os


def run_command(command):
    print(f"\nRunning: {command}\n")
    subprocess.run(command, shell=True, check=True)


# Get video path
video = sys.argv[1]

# Output audio
audio = "audio.wav"

# 1. Extract audio
run_command(
    f'ffmpeg -i "{video}" -vn -ac 1 -ar 16000 "{audio}" -y'
)

# 2. Whisper transcription
run_command(
    f'whisper "{audio}" --model base --language English'
)

# 3. Create chunks
run_command(
    "python chunk_transcript.py"
)

# 4. Create embeddings
run_command(
    "python create_embeddings.py"
)

# 5. Create FAISS index
run_command(
    "python create_faiss_index.py"
)

print("\n✅ Video processing completed!")
print("You can now run: python rag.py")