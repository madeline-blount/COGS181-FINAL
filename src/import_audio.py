import os
import pandas as pd
from pytube import YouTube
from pydub import AudioSegment
import ffmpeg
from tqdm import tqdm

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
AUDIO_PATH = os.path.join(DATA_PATH, "audio")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")

# Ensure directories exist
os.makedirs(AUDIO_PATH, exist_ok=True)

# Load dataset
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed/cleaned_babbling_train_segments.csv"))

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Dataset not found: {csv_path}")

df = pd.read_csv(csv_path)
print("✅ Dataset loaded successfully!")

# Function to download YouTube video
def download_youtube_audio(youtube_id):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    output_path = os.path.join(AUDIO_PATH, f"{youtube_id}.mp4")

    if os.path.exists(output_path):
        print(f"✅ Audio already downloaded: {output_path}")
        return output_path

    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        stream.download(output_path=AUDIO_PATH, filename=f"{youtube_id}.mp4")
        print(f"🎵 Downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"⚠️ Failed to download {url}: {e}")
        return None

# Function to convert MP4 to WAV
def convert_audio(input_path, output_path, start_time, duration):
    if os.path.exists(output_path):
        print(f"✅ Audio already processed: {output_path}")
        return output_path

    try:
        audio = AudioSegment.from_file(input_path, format="mp4")
        trimmed_audio = audio[start_time * 1000:(start_time + duration) * 1000]
        trimmed_audio.export(output_path, format="wav")
        print(f"🔊 Saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"⚠️ Error processing {input_path}: {e}")
        return None

# Process each YouTube ID in the dataset
for index, row in tqdm(df.iterrows(), total=len(df)):
    youtube_id = row["YTID"]
    start_time = float(row["start_seconds"])
    end_time = float(row["end_seconds"])
    duration = end_time - start_time

    # Download audio
    mp4_path = download_youtube_audio(youtube_id)
    if not mp4_path:
        continue

    # Convert & trim audio
    wav_path = os.path.join(AUDIO_PATH, f"{youtube_id}.wav")
    convert_audio(mp4_path, wav_path, start_time, duration)

print("✅ Audio import complete!")