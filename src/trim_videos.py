import pandas as pd
import os
import ffmpeg

# Load cleaned CSV
df = pd.read_csv("../data/processed/cleaned_babbling_train_segments.csv")

# Directory for original and trimmed videos
video_dir = "video_clips"
trimmed_dir = "trimmed_clips"
os.makedirs(trimmed_dir, exist_ok=True)

# Trim each video based on start and end times
for index, row in df.iterrows():
    yt_id = row["YTID"]
    start = row["start_seconds"]
    end = row["end_seconds"]
    
    input_video = os.path.join(video_dir, f"{yt_id}.mp4")
    output_video = os.path.join(trimmed_dir, f"{yt_id}_trimmed.mp4")

    if os.path.exists(input_video):
        ffmpeg.input(input_video, ss=start, to=end).output(output_video).run()
        print(f"Trimmed {yt_id} to {output_video}")

print("Video trimming complete!")