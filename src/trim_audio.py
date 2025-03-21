import pandas as pd
import os
import ffmpeg

# Load cleaned CSV
df = pd.read_csv("../data/processed/cleaned_babbling_train_segments.csv")

# Directory for original and trimmed audio
audio_dir = "audio_clips"
trimmed_audio_dir = "trimmed_audio_clips"
os.makedirs(trimmed_audio_dir, exist_ok=True)

# Trim each audio file based on start and end times
for index, row in df.iterrows():
    yt_id = row["YTID"]
    start = row["start_seconds"]
    end = row["end_seconds"]
    
    input_audio = os.path.join(audio_dir, f"{yt_id}.wav")
    output_audio = os.path.join(trimmed_audio_dir, f"{yt_id}_trimmed.wav")

    if os.path.exists(input_audio):
        ffmpeg.input(input_audio, ss=start, to=end).output(output_audio).run()
        print(f"Trimmed {yt_id} to {output_audio}")

print("Audio trimming complete!")