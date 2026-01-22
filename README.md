PROJECT SYNOPSIS
Title
Emotion Detection System Using Convolutional Neural Networks (CNN)
Introduction
Emotion detection from facial expressions is an important application of computer vision and artificial intelligence. It plays a vital role in areas such as human–computer interaction, mental health analysis, surveillance systems, and interactive applications. This project focuses on building an emotion detection system using deep learning techniques, specifically Convolutional Neural Networks (CNN).
Objectives
To develop an automated system that detects human emotions from facial images.
To understand and implement CNN architecture for image classification.
To train and evaluate the model using a labeled emotion dataset.
To analyze model performance using accuracy metrics.
Scope of the Project
The system is capable of classifying facial images into predefined emotion categories such as happy, sad, angry, fear, surprise, neutral, etc. The project is implemented using Python and TensorFlow/Keras on Google Colab.
Tools and Technologies
Python
TensorFlow / Keras
Google Colab
OpenCV
NumPy & Matplotlib
Methodology (Synopsis)
1.Dataset loading from Google Drive
2.Image preprocessing and augmentation
3.CNN model design
4.Model training and validation
5.Model testing and performance evaluation
Expected Outcome
The system successfully detects emotions from facial images with reasonable accuracy and demonstrates the effectiveness of CNNs in emotion recognition tasks.
FINAL REPORT
Introduction
1.Background
Human emotions play a crucial role in communication. With the advancement of artificial intelligence, machines can now interpret facial expressions to recognize emotions. Emotion detection using deep learning has gained significant attention due to its accuracy and automation capabilities.
2.Problem Statement
Manual analysis of emotions is time-consuming and subjective. There is a need for an automated emotion recognition system that can efficiently analyze facial expressions and classify emotions accurately.
3.Project Motivation
The motivation behind this project is to explore deep learning techniques and apply them to real-world problems such as emotion recognition, enhancing practical understanding of CNNs.
Literature review
Previous studies show that traditional machine learning techniques have limitations in handling complex image data. CNN-based models outperform conventional methods due to automatic feature extraction and hierarchical learning. Recent research emphasizes the use of deep CNNs for emotion recognition with improved accuracy.
Methodology

1.Dataset Description
The dataset is organized into training and testing folders, each containing subfolders representing emotion classes. Images are grayscale facial images resized to a fixed dimension.


2.Dataset Distribution table


3.Data Preprocessing
Image resizing
Normalization
Data augmentation (rotation, zoom, flipping)

4.Model Architecture
The CNN model consists of:
Convolutional layers for feature extraction
Max pooling layers for dimensionality reduction
Dropout layers to reduce overfitting
Fully connected dense layers
Softmax output layer for classification

5.Training Process
The model is trained using categorical cross-entropy loss and Adam optimizer. Training and validation accuracy are monitored to evaluate performance.
6.Evaluation Metrics
Training accuracy
Validation accuracy
Testing accuracy

Implementation details
Complete Source Code (TRAINING)
Below is the complete implementation used for training and testing the Emotion Detection CNN model. The code is executed on Google Colab with Google Drive mounted for dataset access and model storage.
Importing Required Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

Dataset Paths and Parameters
train_dir = '/content/drive/MyDrive/Emotion_Detection_Project/Data/train'
val_dir   = '/content/drive/MyDrive/Emotion_Detection_Project/Data/test'

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
NUM_CLASSES = len(os.listdir(train_dir))

Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

Loading the Dataset
train_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True
)

val_ds = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

CNN Model Architecture
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

Model Compilation and Training along with plot training history
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop, reduce_lr]
)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()


Model Saving
model.save('/content/drive/MyDrive/Emotion_Detection_Project/emotion_cnn_model_final.h5')

Development Environment
Google Colab (GPU enabled)
Python programming language
Libraries Used
TensorFlow
Keras
NumPy
Matplotlib
OpenCV
Model Saved
The trained model is saved in Google Drive for future testing and deployment.









Training Outputs
After training, accuracy was plotted as:



After training loss was plotted as:




Complete Source Code (TESTING)
Importing Required Libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import files

Confusion matrix import
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
	
Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')

Loading trained model
model_path = '/content/drive/MyDrive/Emotion_Detection_Project/emotion_cnn_model_final.h5'
model = load_model(model_path)
print("✅ Model loaded successfully\n")

Dataset paths
train_dir = '/content/drive/MyDrive/Emotion_Detection_Project/Data/train'
test_dir  = '/content/drive/MyDrive/Emotion_Detection_Project/Data/test'

Parameters
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

Loading the Dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    shuffle=False
)

Normalization
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

test_ds = test_ds.map(normalize_img)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

Model evaluation

print("🔹 Evaluating model on full test dataset...\n")
test_loss, test_acc = model.evaluate(test_ds)
print(f"🎯 Test Accuracy: {test_acc*100:.2f}%")
print(f"📉 Test Loss: {test_loss:.4f}\n")

Emotion classes
emotions = sorted(os.listdir(train_dir))
print("Emotion Classes:", emotions, "\n")

Confusion Matrix

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotions)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Emotion Detection")
plt.show()

Classification report

print("📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=emotions))

Image testing

print("🔹 Upload image(s) for prediction")
uploaded = files.upload()

for filename in uploaded.keys():
    img = image.load_img(filename, target_size=IMG_SIZE, color_mode='grayscale')
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_arr)
    predicted_class = emotions[np.argmax(prediction)]
    confidence = np.max(prediction)

    plt.imshow(img_arr[0,:,:,0], cmap='gray')
    plt.title(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

Testing Outputs
After whol dataset tested:

After we uploaded 9 happy images and 8 neutral images:


Results were shown as:
                                           

                                      

                                   

                                    

                            

                            

                             

Confusion Matrix

Classification Report


Results and discussion
The CNN model achieved approximately 70–75% accuracy on training and testing data. The difference between training and testing accuracy indicates controlled overfitting. Performance can be further improved using transfer learning and larger datasets.

Limitations

Limited dataset size
Variations in lighting and facial angles
Lower accuracy for similar emotion classes

Future enhancements

Use of transfer learning models like MobileNet or VGG
Real-time emotion detection using webcam
Larger and more diverse datasets
Integration with mobile or web applications


Conclusion
This project demonstrates the successful implementation of an emotion detection system using CNN. The results confirm that deep learning techniques are effective for facial emotion recognition and provide a foundation for further improvements.
