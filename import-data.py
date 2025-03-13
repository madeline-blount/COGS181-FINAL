import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import re

# Manually define column names
column_names = ['YTID', 'start_seconds', 'end_seconds', 'positive_labels']

# Try reading the file with additional flexibility
df = pd.read_csv("balanced_train_segments.csv", 
                 comment="#", 
                 header=1, 
                 names=column_names, 
                 sep=',',  # Ensure comma as delimiter
                 quotechar='"',  # Handle quoted fields
                 on_bad_lines='skip')  # Skip lines with too many fields

# Clean up any extra spaces in the 'positive_labels' column
df['positive_labels'] = df['positive_labels'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

# Handle NaN values by filling them with an empty string (or you could use another strategy)
df['positive_labels'] = df['positive_labels'].fillna('')

# Show the cleaned data
print(df.head())


