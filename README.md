# 🚗 DentAware – AI-Based Vehicle Damage Classification

**DentAware** is a project that leverages deep learning to classify the type and location of damage on vehicles. Given an image of a car, the model predicts whether it exhibits **breakage**, **crushing**, or **no damage** across both front and rear views.

This tool has practical applications in **insurance automation**, **fleet management**, and **vehicle inspection workflows**, providing faster and more objective damage assessment using AI.

---
## 📹 Demo
https://github.com/user-attachments/assets/aca1611c-fede-403d-9197-a27e2265fab8

---

## 🧠 Project Overview

- 🎯 **Objective**: Automate car damage detection using deep learning
- 📷 **Input**: Car images showing **third-quarter front** or **rear views**
- 🔍 **Output**: One of 6 damage classes with confidence score
- 🛠️ **Solution**: Transfer learning using a fine-tuned ResNet50 model

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
- ⚙️ Served via **FastAPI**, with CORS enabled for frontend access

---

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/dentaware.git
cd dentaware
