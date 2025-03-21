import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build the CNN model for spectrograms
def create_model(input_shape, num_classes):
    model = keras.Sequential([
        # First convolutional layer
        layers.Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2,2)),
        
        # Second convolutional layer
        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),

        # Flatten the output to feed into dense layers
        layers.Flatten(),
        
        # Fully connected dense layer
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model
