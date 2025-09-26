![banner](banner.png)  

# 🚗 DentAware – AI-Based Vehicle Damage Classifier

## Problem Statement
Vehicle damage assessment is traditionally a manual, time-consuming, and error-prone process. Insurance companies, car rental services, and repair shops rely on human evaluators to identify the type and extent of damage on vehicles. 

## 🔎 Project Overview
**DentAware** is a project that leverages deep learning to classify the type and location of damage on vehicles. Given an image of a car, the model predicts whether it exhibits **breakage**, **crushing**, or **no damage** across both front and rear views.

**URL: https://dentaware.streamlit.app**

---

## 📹 Demo Video
https://github.com/user-attachments/assets/af9c2c2e-31dd-476c-85b8-3c0d03976edc

---

## 🧠 Project Overview

- 🎯 **Objective**: Automate car damage detection using deep learning
- 📷 **Input**: Car images showing **third-quarter front** or **rear views**
- 🔍 **Output**: One of 6 damage classes with confidence score
- 🛠️ **Solution**: Transfer learning using a fine-tuned ResNet50 model

---

## 📂 Repository Structure
1. **model** :  A folder containing the saved model
2. **damage_prediction.ipynb** :  jupyter file for protoyping about the project
3. **train.py** :  final reproducable python script to generate trained model
4. **app.py** :  streamlit app code
5. **helper.py** :  the main helper script which takes in new images from the user and classifies it


---

## 🧾 Target Classes

| View        | Damage Type  |
|-------------|--------------|
| Front       | Normal       |
| Front       | Crushed      |
| Front       | Breakage     |
| Rear        | Normal       |
| Rear        | Crushed      |
| Rear        | Breakage     |

---

## 🧪 Model Architecture & Training

- 🧠 **Backbone**: ResNet50 pretrained on ImageNet
- 🔧 **Technique**: Transfer learning with final FC layers replaced and fine-tuned
- 🧹 **Preprocessing**: Image resizing, normalization, RGB conversion
- 📊 **Dataset Size**: ~1,700 labeled images
- 📈 **Validation Accuracy**: ~80% (Stratified split, image augmentation used)

---

## 🔬 Model Deployment

- ✅ Model exported using `torch.save()` and loaded for inference in production
- 🧮 Prediction returns both the **top class** and **confidence score** using `Softmax`
- ⚙️ Built and deployed streamlit UI on render

---
