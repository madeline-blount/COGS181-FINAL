import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import re

# Manually define column names
column_names = ['YTID', 'start_seconds', 'end_seconds', 'positive_labels']

# Read CSV safely, ensuring proper column alignment
df = pd.read_csv("balanced_train_segments.csv", 
                 comment="#", 
                 names=column_names,  # Explicitly assign column names
                 sep=',',  # Force comma as delimiter
                 quotechar='"',  # Handle quoted fields correctly
                 on_bad_lines='skip',  # Skip malformed lines
                 dtype=str,  # Ensure all columns are treated as strings
                 skipinitialspace=True)  # Remove leading/trailing spaces


# Drop any rows where YTID is clearly incorrect (e.g., if it contains numbers)
df = df[df['YTID'].str.match(r'^[a-zA-Z0-9_-]{11}$', na=False)]  

# Ensure YTID is correctly extracted and stripped of spaces
# Clean YTID: Remove any leading dashes and ensure valid YouTube IDs
df['YTID'] = df['YTID'].str.strip().str.lstrip('-')

# Generate YouTube URLs
df['youtube_url'] = "https://www.youtube.com/watch?v=" + df['YTID']

# df.to_csv("cleaned_train_segments.csv", index=False)

# Define the babbling label(s) you want to filter
babbling_labels = ['/m/0261r1']
# Filter rows where 'positive_labels' contains any of the babbling labels
df_babbling = df[df['positive_labels'].apply(lambda x: any(label in x for label in babbling_labels))]

# Verify output
print(df_babbling[['YTID', 'youtube_url']].head())

# Save the filtered data to a new CSV
df_babbling.to_csv("cleaned_babbling_train_segments.csv", index=False)

# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)  # Show all columns

# Check the filtered data
print(df_babbling.head())

