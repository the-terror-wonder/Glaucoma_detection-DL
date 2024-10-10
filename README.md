
# Glaucoma Detection Using ResNet-50

This repository contains a Python notebook that implements a deep learning model for glaucoma detection from retinal fundus images. The model is based on the ResNet-50 architecture and is designed to assist in early glaucoma detection by automatically classifying fundus images.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results](#results)
7. [How to Use](#how-to-use)
8. [Acknowledgments](#acknowledgments)

---

## Project Overview

### Glaucoma and Its Importance
Glaucoma is one of the leading causes of irreversible blindness worldwide. It is characterized by damage to the optic nerve, often associated with elevated intraocular pressure. Early detection and treatment are essential to slow or prevent vision loss. This project aims to build a machine learning model using deep learning techniques to detect glaucoma from retinal fundus images, providing an automated method that can aid healthcare professionals in early diagnosis.

### Objective
The goal of this project is to develop a classification model using a pre-trained ResNet-50 architecture, which will distinguish between glaucomatous and non-glaucomatous fundus images. The model is fine-tuned and trained using transfer learning to achieve high accuracy with relatively limited data.

---

## Installation

### Requirements
The following libraries and tools are required to run the notebook:

- Python 3.x
- TensorFlow 2.x / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV (for image processing)
- Jupyter Notebook (for running the `.ipynb` file)

You can install the required libraries using the following command:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python jupyter
```

### Clone the Repository
To get started, clone this repository to your local machine:
```bash
git clone <repository-url>
cd <repository-directory>
```

---

## Data Preparation

### Dataset
The project uses a publicly available dataset of retinal fundus images. You need to ensure that the dataset is structured properly with two folders:
- **Glaucoma**: Images labeled as glaucomatous.
- **Non-Glaucoma**: Images labeled as non-glaucomatous.

If your dataset is in a different format, you may need to modify the data loading functions in the notebook accordingly.

### Preprocessing
Before feeding the images into the model, several preprocessing steps are applied:
- **Resizing**: All images are resized to a standard input size of 224x224 pixels, as required by ResNet-50.
- **Normalization**: Pixel values are normalized to improve convergence during training.
- **Data Augmentation**: Techniques like rotation, flipping, zooming, and shifting are applied to artificially increase the size of the dataset and prevent overfitting.

---

## Model Architecture

The model is based on the **ResNet-50** architecture, a deep convolutional neural network that is 50 layers deep. ResNet-50 is a popular choice for image classification tasks due to its ability to efficiently train very deep networks using residual connections.

### Transfer Learning
A pre-trained ResNet-50 model, trained on the ImageNet dataset, is used as the base. Transfer learning is applied, where the fully connected layers are replaced with a new classifier specific to glaucoma detection. The following modifications are made:
- The top layers of the ResNet-50 model are removed.
- New fully connected layers and a softmax classifier are added for binary classification (glaucoma/non-glaucoma).
- The lower layers of ResNet-50 are frozen during initial training to retain the general image features learned from ImageNet.

---

## Training and Evaluation

### Training
The model is trained using a combination of categorical cross-entropy loss and the Adam optimizer. Key training strategies include:
- **Fine-tuning**: After initial training, some of the lower layers of ResNet-50 are unfrozen for fine-tuning to adapt the model more specifically to the glaucoma detection task.
- **Early Stopping**: Early stopping is employed to prevent overfitting by stopping the training process once the validation accuracy stops improving.

### Evaluation
The trained model is evaluated using the following metrics:
- **Accuracy**: The proportion of correct predictions.
- **Precision and Recall**: To account for class imbalance, precision and recall are calculated to measure the model's ability to detect true glaucoma cases.
- **Confusion Matrix**: A confusion matrix is plotted to visualize the model's performance in distinguishing between glaucoma and non-glaucoma images.

---

## Results

### Model Performance
After training, the model achieves competitive results in glaucoma detection. The results will vary based on the dataset and the hyperparameters chosen, but typical performance metrics include:
- **Accuracy**: 90%+
- **Precision**: High precision in detecting glaucoma cases.
- **Recall**: Good recall, minimizing false negatives.

The confusion matrix and loss/accuracy curves during training are visualized in the notebook for further insights into the model's behavior.

---

## How to Use

### Running the Notebook
1. Ensure you have the required dependencies installed.
2. Open the notebook:
   ```bash
   jupyter notebook glaucoma_detection_resnet50.ipynb
   ```
3. Follow the steps in the notebook:
   - Load and preprocess the dataset.
   - Train the model or load pre-trained weights if available.
   - Evaluate the model on a test set.

### Customize for Your Dataset
If you have a different dataset, ensure that it follows the correct directory structure, and update the data loading paths in the notebook. The notebook is designed to be easily adaptable to new datasets and use cases.

---

## Acknowledgments

This project is based on the ResNet-50 architecture, pre-trained on ImageNet. Special thanks to the creators of the dataset used for training and evaluation. The open-source community and the creators of TensorFlow, Keras, and other essential libraries are gratefully acknowledged.
