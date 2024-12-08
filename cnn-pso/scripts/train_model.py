# Import necessary libraries
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pyswarms as ps
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Load the Breast Cancer Ultrasound dataset
data_dir = '../data/train'  # Update with the path to your dataset
images = []
labels = []

# Load images and labels
print("Loading images...")
for label in ['Normal', 'Benign', 'Malignant']:
    folder_path = os.path.join(data_dir, label)
    print(f"Processing folder: {folder_path}")
    for img_file in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, img_file))
        if img is not None:  # Check if the image was loaded successfully
            img = cv2.resize(img, (128, 128))  # Resize images to 128x128
            images.append(img)
            labels.append(label)
        else:
            print(f"Warning: Could not load image {img_file}")

# Convert to numpy arrays
images = np.array(images, dtype='float32') / 255.0  # Normalize pixel values
labels = pd.get_dummies(labels).values  # One-hot encode labels

# Print the number of images loaded
print(f"Loaded {len(images)} images with shape {images[0].shape} if images else 'No images loaded.'")

# Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)
print(f"Training set size: {x_train.shape[0]}, Testing set size: {x_test.shape[0]}")

# Data augmentation
print("Setting up data augmentation...")
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
datagen.fit(x_train)
print("Data augmentation setup complete.")

# Define CNN model architecture
def create_cnn_model():
    print("Creating CNN model...")
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(layers.Dense(3, activation='softmax'))  # Three classes: normal, benign, malignant
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("CNN model created and compiled.")
    return model

# Implement PSO for hyperparameter optimization
def fitness_function(params):
    learning_rate = params[:, 0]
    batch_size = params[:, 1].astype(int)
    
    accuracies = []
    print("Evaluating fitness function...")
    for lr, bs in zip(learning_rate, batch_size):
        print(f"Testing with learning rate: {lr}, batch size: {bs}")
        model = create_cnn_model()
        model.optimizer.learning_rate = lr
        model.fit(datagen.flow(x_train, y_train, batch_size=bs), epochs=5, verbose=0)  # Reduced epochs for testing
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        accuracies.append(accuracy)
        print(f"Accuracy for lr={lr}, bs={bs}: {accuracy}")
    
    return -np.array(accuracies)  # Minimize negative accuracy

# Define bounds for learning rate and batch size
bounds = (np.array([0.0001, 16]), np.array([0.1, 128]))

# Create a PSO optimizer
print("Creating PSO optimizer...")
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=2, options={'c1': 0.5, 'c2': 0.5, 'w': 0.9}, bounds=bounds)

# Perform optimization
print("Starting optimization...")
best_cost, best_pos = optimizer.optimize(fitness_function, iters=20)  # Increased iterations
print(f"Best cost: {best_cost}, Best position: {best_pos}")

# Train the model with the best hyperparameters
best_learning_rate = best_pos[0]
best_batch_size = int(best_pos[1])
print(f"Best learning rate: {best_learning_rate}, Best batch size: {best_batch_size}")

cnn_model = create_cnn_model()
cnn_model.optimizer.learning_rate = best_learning_rate

# Callbacks for early stopping and model checkpointing
checkpoint = ModelCheckpoint('../model/best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model with the best hyperparameters
print("Starting model training...")
cnn_model.fit(datagen.flow(x_train, y_train, batch_size=best_batch_size), 
               epochs=20, 
               validation_data=(x_test, y_test), 
               callbacks=[checkpoint, early_stopping], 
               verbose=1)  # Added verbose output

# Save the trained model ][0oitrewq]
cnn_model.save('../model/breast_cancer_cnn_model.keras')
print("Model training complete and saved.")

# Optional: Visualize some predictions
predictions = cnn_model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Display a few test images with their predicted and true labels
print("Displaying predictions for a few test images...")
for i in range(5):
    plt.imshow(x_test[i])
    plt.title(f"Predicted: {predicted_classes[i]}, True: {true_classes[i]}")
    plt.axis('off')
    plt.show()
