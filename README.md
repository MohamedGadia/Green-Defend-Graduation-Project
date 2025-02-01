## Green Defend-CNN

**Graduation Project:** *Green Defend - Intelligent Plant Disease Detection and Agriculture Support System*  
**Institute:** Mansoura Higher Institute of Engineering and Technology (Engineering College)

- A machine learning model leveraging a CNN to achieve 98.7% accuracy in classifying 38 plant diseases 
from over 87,000 images.
- The dataset was split into 56,251 images for training, 14,044 images for validation, and 17,572 images 
for testing.
- Designed a multi-layer CNN architecture with ELU activations, Dropout, and GlorotNormal initialization 
to optimize performance.
- Utilized TensorFlow and Keras libraries, incorporating techniques like Early Stopping and Model 
Checkpoint to prevent overfitting and optimize performance.
- Evaluated using precision, recall, F1-score, and confusion matrix.
- Integrated the model into a mobile and web application for real-time disease detection.
- Earned an A+ grade

---

## Project Overview

The goal of this project is to identify 38 different plant diseases from RGB images of crop leaves. By leveraging deep learning techniques, this model helps in detecting and categorizing diseases, aiding farmers and agricultural experts.

### Dataset
- **Source**: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
- **Statistics**:
  - Training Data: 56,251 images( 64.7% of the data )
  - Validation Data: 14,044 images( 16.1% of the data )
  - Test Data: 17,572 images( 20.2% of the data )
  - Classes: 38 (including healthy and diseased leaves)

---

## Key Features

### 1. Data Preprocessing
- Images are loaded directly from their directories using `ImageDataGenerator`, without the need for data augmentation.
- Only **rescaling** of pixel values is applied during loading to normalize the data.

### 2. Exploratory Data Analysis (EDA)
- Insights into dataset composition (e.g., class distribution, number of unique plants).
- Visualization of sample images and disease categories.

### 3. CNN Architecture
- Multiple Conv2D layers with ELU activations and GlorotNormal kernel initialization.
- Pooling, Dropout, and Dense layers to optimize model performance.
- Input shape: `(224, 224, 3)`

### 4. Model Training
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Callbacks:
  - `ModelCheckpoint`: Saves the best-performing model.
  - `EarlyStopping`: Stops training to prevent overfitting.

### 5. Evaluation
- Test accuracy and loss reported.
- Classification report.

### 6. Predictions
- Visualization of model predictions on random test samples.

---

## TensorFlow Lite (TFLite) Model for Mobile Application
**File Name**: `green_defend_cnn.tflite`
#### **TensorFlow Lite (TFLite)** version of the trained model was created to facilitate its use in a mobile application. This optimized version ensures efficient, real-time plant disease detection directly on mobile devices.


## Results
- **Train Accuarcy**: Achieved high accuracy during training, demonstrating the model's ability to learn effectively from the data.

- **Validation Accuracy**:Consistently aligned with training accuracy, indicating that the model generalizes well without overfitting.

- **Test Accuracy**: The model achieved an outstanding **98.7%** accuracy on the test dataset, demonstrating its robustness in correctly classifying plant diseases across unseen data.
  
- **Key Metrics**: Classification report includes precision, recall, and F1-score for each class.

---

### Key Libraries:
- TensorFlow(with Keras)
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
---
### Link NoteBook in Kaggle: [Green-Defend-CNN](https://www.kaggle.com/code/mohamedahmedgadia/green-defend-cnn?scriptVersionId=210199855) 

---

## Programmer
**Mohamed Ahmed Gadia**  
Email: [mohamedgadia00@gmail.com]  
LinkedIn: [Mohamed Gadia](https://www.linkedin.com/in/mohamedgadia) 
