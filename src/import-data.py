import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import re

# Manually define column names
column_names = ['YTID', 'start_seconds', 'end_seconds', 'positive_labels']

# Get the absolute path of the file
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/balanced_train_segments.csv"))

# Ensure the file exists before reading
if not os.path.exists(file_path):
    raise FileNotFoundError(f" CSV file not found: {file_path}")

# Read CSV with proper column handling
df = pd.read_csv(
    file_path, 
    comment="#", 
    names=column_names,  # Explicitly assign column names
    sep=',',  # Force comma as delimiter
    quotechar='"',  # Handle quoted fields correctly
    on_bad_lines='skip',  # Skip malformed lines
    dtype=str,  # Ensure all columns are treated as strings
    skipinitialspace=True  # Remove leading/trailing spaces
)

print(" File loaded successfully!")

# Drop invalid YouTube IDs (Only allow valid 11-character YouTube video IDs)
df = df[df['YTID'].str.match(r'^[a-zA-Z0-9_-]{11}$', na=False)]

# Ensure YTID is correctly stripped of spaces and leading dashes
df['YTID'] = df['YTID'].str.strip().str.lstrip('-')

# Generate YouTube URLs
df['youtube_url'] = "https://www.youtube.com/watch?v=" + df['YTID']

# Define the babbling label(s) you want to filter
babbling_labels = ['/m/0261r1']

# Handle missing values before applying the filter
df['positive_labels'] = df['positive_labels'].fillna("")

# Filter rows where 'positive_labels' contains any of the babbling labels
df_babbling = df[df['positive_labels'].apply(lambda x: any(label in x for label in babbling_labels))]

# Verify output
print("📝 Filtered dataset (babbling videos):")
print(df_babbling[['YTID', 'youtube_url']].head())

# Ensure the output directory exists
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed/"))
os.makedirs(output_dir, exist_ok=True)

# Save the filtered data to a new CSV in the processed folder
output_file = os.path.join(output_dir, "cleaned_babbling_train_segments.csv")
df_babbling.to_csv(output_file, index=False)
print(f" Cleaned dataset saved to: {output_file}")

# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)

# Check the filtered data
print(df_babbling.head())