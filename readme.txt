# ♻️ AI Recycling Image Classifier

This project implements a real-time image classification system for recyclable waste using deep learning. It is based on **ResNet101** and applies **transfer learning** techniques to achieve high accuracy in classifying five waste categories: **glass**, **plastic**, **paper**, **metal**, and **household garbage**.

---

## 📌 Project Overview

The goal of this project is to support smart recycling by automating the detection and classification of waste from images. The system is built with the following key features:
- Fine-tuned **ResNet101 CNN** for image classification
- Custom dataset of labeled waste images
- Desktop interface for real-time predictions using live video (IP Webcam)
- Optimized for a balance of **accuracy**, **speed**, and **ease of use**

---

## 📊 Model Performance

- Accuracy: **93.48%** on test data  
- Optimizer: `AdamW`  
- Techniques: Dropout, Data Augmentation, Manual Labeling  
- Training done on **Google Colab GPU (T4)**

---

AI-Recycling-Image-Classifier/
│
├── Interface/ # Desktop interface for classification
├── ResNet101-Model/ # Training and evaluation code
├── README.md

---

## 🔗 Resources

| Resource                          | Link |
|----------------------------------|------|
| 📁 Dataset Structure             | [Google Drive Folder](https://drive.google.com/drive/folders/1iltNpc8JVqZkF6RR7nXjmjf311SCzpdp?usp=sharing) |
| 🧠 Trained Model (ResNet101)     | [Colab Notebook - Model Only](https://colab.research.google.com/drive/1cv4v0lmAzPOEhovvlcB3GKObS92tF5E6?usp=sharing) |
| 💻 Full Source Code & Training   | [Colab Notebook - Full Model Code](https://colab.research.google.com/drive/1SVe63kOiQAslGdfzQ-X53x3HM7Il6ECS?usp=sharing) |

---

## 🛠️ Technologies Used

- Python 3.10
- PyTorch
- OpenCV
- Google Colab
- Tkinter (or PyQt)
- IP Webcam (for mobile video input)

---

## 📦 Installation

```bash
git clone https://github.com/NiroIf91/AI-Recycling-Image-Classifier.git
cd AI-Recycling-Image-Classifier
pip install -r requirements.txt
