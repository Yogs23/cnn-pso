# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Path to your dataset
test_dir = '../data/test'  # Ganti dengan path ke dataset Anda

# ImageDataGenerator for data loading and augmentation
datagen = ImageDataGenerator(rescale=1./255)

# Load the test data with the correct target size
test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),  # Sesuaikan dengan ukuran input model Anda
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Penting untuk menjaga urutan untuk evaluasi
)

# Load the trained model
model_path = '../model/breast_cancer_cnn_model.keras'  # Ganti dengan path ke model yang disimpan
print(f"Loading model from {model_path}...")
cnn_model = load_model(model_path)

# Print model summary to verify input/output shapes
cnn_model.summary()

# Evaluate the model
print("Evaluating the model...")
test_gen.reset()  # Reset the generator untuk memastikan prediksi dibuat pada data yang benar
predictions = cnn_model.predict(test_gen)  # Gunakan semua gambar untuk evaluasi
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes

# Generate classification report
print("Generating classification report...")
class_labels = list(test_gen.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Predict on some sample images from test set
sample_images, _ = next(test_gen)
predictions = cnn_model.predict(sample_images[:5])  # Prediksi pada 5 gambar pertama
predicted_classes = np.argmax(predictions, axis=1)

# Display sample images with predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(sample_images[i])
    ax.set_title(f"Predicted: {class_labels[predicted_classes[i]]}")
    ax.axis("off")
plt.show()
