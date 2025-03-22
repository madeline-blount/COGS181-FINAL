'''
import numpy as np
from data_processing import load_and_process_csv, process_spectrograms_and_labels, split_and_preprocess_data
from cnn_model import create_model

# Paths
spectrogram_dir = "trimmed_spectrograms"
csv_path = "../data/processed/cogs181_video_annotations2.0.csv"
label_mapping = {'Happy': 0, 'Neutral': 1, 'Distressed': 2, 'Other': 3}

# Load and preprocess data
df = load_and_process_csv(csv_path, label_mapping)
X, y = process_spectrograms_and_labels(spectrogram_dir, df)

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

'''

'''
import numpy as np
from data_processing import load_and_process_csv, process_spectrograms_and_labels, split_and_preprocess_data
from cnn_model import create_model

# Paths
spectrogram_dir = "trimmed_spectrograms"
csv_path = "../data/processed/cogs181_video_annotations2.0.csv"
label_mapping = {'Happy': 0, 'Neutral': 1, 'Distressed': 2, 'Other': 3}

# Load and preprocess data
df = load_and_process_csv(csv_path, label_mapping)
X, y = process_spectrograms_and_labels(spectrogram_dir, df)

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
print(f"Test Accuracy: {test_acc:.2f}")

# --------------------------
# 🔹 Modify Predictions for Controlled Accuracy
# --------------------------

def modify_predictions(y_true, accuracy_target):
    """
    Modifies predictions to achieve a specific accuracy.
    :param y_true: True labels
    :param accuracy_target: Target accuracy (e.g., 0.25, 0.5, 0.75)
    :return: Modified predictions
    """
    n_samples = len(y_true)
    correct_predictions = int(n_samples * accuracy_target)
    incorrect_predictions = n_samples - correct_predictions

    # Generate random indices for incorrect predictions
    incorrect_indices = np.random.choice(n_samples, incorrect_predictions, replace=False)

    # Copy true labels
    y_pred = np.copy(y_true)

    # Modify incorrect predictions randomly
    for i in incorrect_indices:
        possible_labels = list(set(y_true) - {y_true[i]})  # Choose wrong label
        y_pred[i] = np.random.choice(possible_labels)

    return y_pred

# --------------------------
# 🔹 Simulate Different Accuracy Levels
# --------------------------
for target_accuracy in [0.25, 0.5, 0.75]:
    simulated_y_pred = modify_predictions(y_test, target_accuracy)
    simulated_accuracy = np.mean(simulated_y_pred == y_test)

    print(f"Simulated Test Accuracy: {simulated_accuracy:.2f} (Target: {target_accuracy})")

'''


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
num_classes = 4
num_samples = 1000  # Number of samples to simulate
target_accuracies = [0.25, 0.5, 0.75]  # Accuracy levels to simulate

# --- Generate true labels ---
y_true = np.random.randint(0, num_classes, size=num_samples)

# --- Helper: Simulate predictions at given accuracy ---
def simulate_predictions(y_true, accuracy, num_classes):
    y_pred = []
    for true_label in y_true:
        if np.random.rand() < accuracy:
            y_pred.append(true_label)  # correct prediction
        else:
            other_labels = list(set(range(num_classes)) - {true_label})
            y_pred.append(np.random.choice(other_labels))
    return np.array(y_pred)

# --- Simulate for each accuracy level ---
for acc in target_accuracies:
    print(f"\n📊 Simulated Accuracy: {acc}")
    y_pred = simulate_predictions(y_true, acc, num_classes)

    # Show classification report
    print(classification_report(y_true, y_pred, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title(f"Confusion Matrix (Simulated Accuracy: {acc})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
