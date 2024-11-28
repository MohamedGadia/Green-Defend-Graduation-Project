## Green Defend-CNN

**Graduation Project:** *Green Defend - Intelligent Plant Disease Detection and Agriculture Support System*  
**Institute:** Mansoura Higher Institute of Engineering and Technology (Engineering College)

- Developed a mobile and web application leveraging artificial intelligence to detect plant diseases with an accuracy of 99%.
- Built a model based on Convolutional Neural Networks (CNN) to classify 38 plant diseases using over 87,000 images.
- Successfully integrated the model into a mobile and web application, providing a practical and user-friendly solution for farmers and agricultural experts.
- Had an - grade

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

- **Test Accuracy**: The model achieved an outstanding **99%** accuracy on the test dataset, demonstrating its robustness in correctly classifying plant diseases across unseen data.
  
- **Key Metrics**: Classification report includes precision, recall, and F1-score for each class.

---

### Key Libraries:
- TensorFlow(with Keras)
- Pandas
- NumPy
- Matplotlib
- Seaborn
---
### Link NoteBook in Kaggle: [Green-Defend-CNN](https://www.kaggle.com/code/mohamedahmedgadia/green-defend-cnn?scriptVersionId=210152502) 

---

## Programmer
**Mohamed Ahmed Gadia**  
Email: [mohamedgadia00@gmail.com]  
LinkedIn: [Mohamed Gadia](https://www.linkedin.com/in/mohamedgadia) 
