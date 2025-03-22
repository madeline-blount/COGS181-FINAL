import pandas as pd
import os
import subprocess

# Load cleaned CSV
df = pd.read_csv("../data/processed/cleaned_babbling_train_segments.csv")

# Directory to save videos
video_dir = "video_clips"
os.makedirs(video_dir, exist_ok=True)

# Download each video
for yt_id in df["YTID"]:
    video_url = f"https://www.youtube.com/watch?v={yt_id}"
    output_path = os.path.join(video_dir, f"{yt_id}.mp4")

    # Check if file exists to avoid re-downloading
    if not os.path.exists(output_path):
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",  # Best quality available
            "--merge-output-format", "mp4",
            "-o", output_path,
            video_url
        ]
        subprocess.run(cmd)

print("Video download complete!")