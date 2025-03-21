import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import json

def load_and_process_csv(csv_path, label_mapping):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Function to extract emotion labels from the JSON strings
    def extract_emotions(json_string):
        try:
            emotion_data = json.loads(json_string)
            emotions = [emotion["timelinelabels"][0] for emotion in emotion_data]
            return emotions
        except Exception as e:
            print(f"Error parsing emotion labels: {e}")
            return []

    df['emotion_labels'] = df['emotion_labels'].apply(extract_emotions)

    # Map the emotion labels to integers
    df['emotion_labels'] = df['emotion_labels'].apply(
        lambda x: [label_mapping[label] for label in x if label in label_mapping]
    )

    df = df.explode('emotion_labels')
    return df

def process_spectrograms_and_labels(spectrogram_dir, df):
    X = []
    y = []
    missing_samples = 0

    for filename in os.listdir(spectrogram_dir):
        if filename.endswith(".png"):
            file_id = filename.replace("_trimmed.png", "")
            label_row = df[df["video"].str.contains(file_id, case=False, na=False)]
            
            if label_row.empty:
                missing_samples += 1
                print(f"No matching label found for file: {file_id}")
            else:
                spectrogram_path = os.path.join(spectrogram_dir, filename)
                spectrogram = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)
                spectrogram = cv2.resize(spectrogram, (128, 128))
                spectrogram = spectrogram / 255.0

                X.append(spectrogram)
                y.append(label_row["emotion_labels"].values[0])

    print(f"Missing samples: {missing_samples}")
    X = np.array(X).reshape(-1, 128, 128, 1)
    y = np.array(y)
    
    return X, y


def process_mfccs_and_labels(mfcc_dir, df):
    X = []
    y = []
    missing_samples = 0

    # Loop through the MFCC images
    for filename in os.listdir(mfcc_dir):
        if filename.endswith(".png"):  # Assuming MFCC images are also saved as .png
            file_id = filename.replace("_trimmed.png", "")  # Adjust according to your filename pattern
            label_row = df[df["video"].str.contains(file_id, case=False, na=False)]
            
            if label_row.empty:
                missing_samples += 1
                print(f"No matching label found for file: {file_id}")
            else:
                mfcc_path = os.path.join(mfcc_dir, filename)
                mfcc_image = cv2.imread(mfcc_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale image

                # Resize to the input shape expected by the CNN
                mfcc_image = cv2.resize(mfcc_image, (128, 128))  # Resize to match your model input
                mfcc_image = mfcc_image / 255.0  # Normalize the pixel values to [0, 1]

                X.append(mfcc_image)
                y.append(label_row["emotion_labels"].values[0])  # Ensure this points to the correct label column

    print(f"Missing samples: {missing_samples}")
    X = np.array(X).reshape(-1, 128, 128, 1)  # Add the channel dimension
    y = np.array(y)

    return X, y


def split_and_preprocess_data(X, y, num_classes):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test