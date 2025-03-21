import os
import pandas as pd
import yt_dlp
from pydub import AudioSegment

# Load your dataset
df_babbling = pd.read_csv("../data/processed/cleaned_babbling_train_segments.csv", dtype=str)
print(df_babbling.head())  # Verify that the data loads correctly

# Make sure directory exists
output_dir = "audio_clips"
os.makedirs(output_dir, exist_ok=True)

# Function to download and trim audio
def download_audio(youtube_id, start_time, end_time, output_dir):
    """Download YouTube audio and trim it."""
    
    output_path = os.path.join(output_dir, f"{youtube_id}.wav")

    # Check if file already exists to avoid re-downloading
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping download.")
        return
    
    # Download audio using yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, f"{youtube_id}.%(ext)s"),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
        
        # Load the downloaded audio
        audio = AudioSegment.from_file(os.path.join(output_dir, f"{youtube_id}.wav"))
        
        # Trim to the correct segment
        trimmed_audio = audio[start_time * 1000 : end_time * 1000]  # Convert seconds to milliseconds
        trimmed_audio.export(output_path, format="wav")
        print(f"Downloaded and trimmed: {output_path}")
    
    except Exception as e:
        print(f"Failed to download {youtube_id}: {e}")

# Loop through dataset and download clips
for _, row in df_babbling.iterrows():
    download_audio(row['YTID'], row['start_seconds'], row['end_seconds'], output_dir)