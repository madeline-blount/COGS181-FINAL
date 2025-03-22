import numpy as np
from data_processing import load_and_process_csv, process_mfccs_and_labels, split_and_preprocess_data
from cnn_model import create_model

# Paths
mfcc_dir = "trimmed_mfccs"
csv_path = "../data/processed/cogs181_video_annotations2.0.csv"
label_mapping = {'Happy': 0, 'Neutral': 1, 'Distressed': 2, 'Other': 3}

# Load and preprocess data
df = load_and_process_csv(csv_path, label_mapping)
X, y = process_mfccs_and_labels(mfcc_dir, df)

# Split and preprocess data
num_classes = len(label_mapping)
X_train, X_val, X_test, y_train, y_val, y_test = split_and_preprocess_data(X, y, num_classes)

# Check shapes of y_train, y_val, and y_test
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"y_test shape: {y_test.shape}")

# Convert y_train, y_val, y_test back to integer labels
y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test, axis=1)

print(f"y_train shape after conversion: {y_train.shape}")
print(f"y_val shape after conversion: {y_val.shape}")
print(f"y_test shape after conversion: {y_test.shape}")

# Define the CNN model
input_shape = (128, 128, 1)
model = create_model(input_shape, num_classes)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")