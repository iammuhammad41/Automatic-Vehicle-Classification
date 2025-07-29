# Vehicle Classification Using Deep Learning

## Overview

This project focuses on vehicle classification using deep learning techniques. It leverages popular pre-trained models such as ResNet50, VGG16, and MobileNetV2, alongside a custom CNN model for classifying vehicles into various categories. The dataset consists of images from different types of vehicles, including SUVs, buses, fire engines, taxis, and more.

### Objective

The primary goal is to classify images of vehicles into predefined categories using deep learning models. The project demonstrates how to:

* Preprocess image data.
* Implement a custom CNN model.
* Fine-tune pre-trained models (ResNet50, VGG16, and MobileNetV2).
* Train and evaluate the models using PyTorch.

## Features

1. **Custom CNN Architecture**: A custom-built convolutional neural network (CNN) is used for classification.
2. **Transfer Learning**: Pre-trained models (ResNet50, VGG16, MobileNetV2) are fine-tuned to improve classification performance.
3. **Data Augmentation**: Images are augmented to increase model robustness (e.g., random rotations, flips).
4. **Model Evaluation**: Models are evaluated using accuracy, loss, and visualized training curves.

## Installation

To run this code, make sure to have the following dependencies installed:

```bash
pip install torch torchvision matplotlib scikit-learn pandas kagglehub
```

## Dataset

This project uses the vehicle classification dataset provided in the Kaggle competition "Vehicle Classification". The dataset contains images of various vehicle types categorized as:

* SUV
* Bus
* Family Sedan
* Fire Engine
* Heavy Truck
* Jeep
* Minibus
* Racing Car
* Taxi
* Truck

### Dataset Directory Structure

```
dataset/
│
├── train/           # Training images
├── val/             # Validation images
└── test/            # Test images
```

## Code Structure

### 1. **Data Preprocessing**

The dataset is loaded using the `pandas` library and organized into training, validation, and test sets. Images are preprocessed by resizing and normalizing them using `torchvision.transforms`.

### 2. **Model Architecture**

The following models are used:

* **MyCNN**: A custom CNN model with four convolutional blocks, batch normalization, and fully connected layers for classification.
* **ResNet50**: A pre-trained ResNet50 model with a modified final layer for vehicle classification.
* **VGG16**: A pre-trained VGG16 model with a modified final layer.
* **MobileNetV2**: A pre-trained MobileNetV2 model with a modified final layer.

Each model is trained using the AdamW optimizer with a learning rate scheduler (CosineAnnealingLR).

### 3. **Training**

The training loop involves:

* Loading the images and labels in batches using `DataLoader`.
* Computing the loss using `CrossEntropyLoss` and updating the model's weights with backpropagation.
* Monitoring validation accuracy and saving the best model based on validation performance.

### 4. **Model Evaluation**

The models are evaluated on the test set using accuracy and loss metrics. The prediction results for a few test images are visualized, and the class labels are printed.

## Example Usage

1. **Train a model:**
   To train the custom CNN model, ResNet50, VGG16, or MobileNetV2 on the dataset, run the following:

   ```python
   # For custom model
   model = MyCNN().to(device)
   train_model(model, 'my_model', train_loader, val_loader)

   # For ResNet50
   model = resnet50(weights=None).to(device)
   train_model(model, 'resnet', train_loader, val_loader)

   # For VGG16
   model = vgg16(weights=None).to(device)
   train_model(model, 'vgg16', train_loader, val_loader)

   # For MobileNetV2
   model = mobilenet_v2(weights=None).to(device)
   train_model(model, 'mobilenet', train_loader, val_loader)
   ```

2. **Predict the class of an image:**
   Use the trained model to predict the class of a new image:

   ```python
   test_image_path = "/path/to/test/image.jpg"  # Replace with the actual image path
   predicted_class = predict_image(test_image_path, my_model, CLASS_NAMES)
   print(f"Predicted Class: {predicted_class}")
   ```

### Example output:

```bash
Predicted Class: SUV
```

## Results

After training the models, the following results are produced:

* Training and validation accuracy curves.
* Best performing model (based on validation accuracy) is saved for later use.
* Test set predictions are evaluated and displayed.
