# DentAware – AI-Based Vehicle Damage Classifier

**DentAware** leverages deep learning to classify and locate damage on vehicles from images, predicting **breakage**, **crushing**, or **no damage** across front and rear views.

## Project URL
Visit the [DentAware Web App](https://dentaware.streamlit.app) to try it out live.

## Demo Video
Watch our demo video for a visual walkthrough: [DentAware Demo](https://github.com/user-attachments/assets/af9c2c2e-31dd-476c-85b8-3c0d03976edc)

---

## Project Overview

- **Objective**: Automate car damage detection using deep learning.
- **Input**: Car images of third-quarter front or rear views.
- **Output**: Classification into one of 6 damage categories with confidence scores.
- **Solution**: Transfer learning with a fine-tuned ResNet50 model.

---

## Target Classes

| View        | Damage Type  |
|-------------|--------------|
| Front       | Normal       |
| Front       | Crushed      |
| Front       | Breakage     |
| Rear        | Normal       |
| Rear        | Crushed      |
| Rear        | Breakage     |

---

## Model Architecture & Training

- **Backbone**: ResNet50 pretrained on ImageNet.
- **Technique**: Transfer learning with custom FC layers, fine-tuned on a specific car damage dataset.
- **Preprocessing**: Image resizing, normalization, RGB conversion.
- **Dataset Size**: ~1,700 labeled images.
- **Validation Accuracy**: ~80% (Stratified split, image augmentation used).

---

## Model Deployment

- **Deployment**: Model exported using `torch.save()` for production inference.
- **Prediction**: Returns top class and confidence score using `Softmax`.
- **User Interface**: Implemented using Streamlit for intuitive interaction.

---

This README provides an overview of DentAware, detailing its objectives, model architecture, training approach, and deployment strategy.
