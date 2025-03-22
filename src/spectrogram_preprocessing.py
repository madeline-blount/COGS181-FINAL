import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import noisereduce as nr
import webrtcvad

# Define directories
audio_dir = "audio_clips"
processed_dir = "processed_audio"
spectrogram_dir = "spectrograms"
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(spectrogram_dir, exist_ok=True)

def apply_vad(audio_data, sample_rate):
    # Resample to 16 kHz and ensure mono channel
    audio_data_resampled = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
    
    # Convert audio to 16-bit PCM (int16) for VAD
    audio_data_resampled = (audio_data_resampled * 32767).astype(np.int16)
    
    # Initialize WebRTC VAD (mode 1 is mild aggressiveness)
    vad = webrtcvad.Vad(1)
    
    # Set frame size (20 ms frames at 16 kHz)
    frame_duration = 20  # in milliseconds
    frame_size = int(16000 * frame_duration / 1000)  # 160 samples for 16 kHz

    # Ensure that the number of samples in the audio is divisible by frame_size
    if len(audio_data_resampled) % frame_size != 0:
        padding_length = frame_size - (len(audio_data_resampled) % frame_size)
        audio_data_resampled = np.pad(audio_data_resampled, (0, padding_length), 'constant')

    # Split the audio into frames (each frame has `frame_size` samples)
    frames = [audio_data_resampled[i:i + frame_size] for i in range(0, len(audio_data_resampled), frame_size)]

    # Apply VAD to each frame (check if it's speech)
    speech_frames = []
    for frame in frames:
        # Convert frame to bytes and apply VAD
        if vad.is_speech(frame.tobytes(), 16000):
            speech_frames.append(frame)
    
    # Combine speech frames into one array
    vad_audio = np.concatenate(speech_frames)

    # Convert the resulting audio back to float32
    vad_audio = vad_audio.astype(np.float32) / 32767.0  # Normalize back to float range

    return vad_audio

def pad_audio(y, target_length):
    """Pad audio to match original length."""
    pad_length = target_length - len(y)
    if pad_length > 0:
        y = np.pad(y, (0, pad_length), 'constant')
    elif pad_length < 0:
        y = y[:target_length]  # Trim if necessary
    return y

def process_audio(file_path, output_path, sr=16000):
    """Load, apply VAD, pad, and save processed audio."""
    y, sr = librosa.load(file_path, sr=sr)
    original_length = len(y)

    # Apply VAD
    vad_audio = apply_vad(y, sr)

    # Pad back to original length
    padded_audio = pad_audio(vad_audio, original_length)

    # Save processed audio
    sf.write(output_path, padded_audio, sr)
    return padded_audio, sr

def save_spectrogram(y, sr, output_path):
    """Convert audio to spectrogram and save as an image."""
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(5, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.savefig(output_path)
    plt.close()

# Process all audio files
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        input_path = os.path.join(audio_dir, filename)
        output_audio_path = os.path.join(processed_dir, filename)
        output_spectrogram_path = os.path.join(spectrogram_dir, filename.replace(".wav", ".png"))

        print(f"Processing {filename}...")
        processed_audio, sr = process_audio(input_path, output_audio_path)
        save_spectrogram(processed_audio, sr, output_spectrogram_path)

print("Preprocessing complete!")

