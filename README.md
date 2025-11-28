
<!-- ![Project Banner](docs/banner.svg) -->

# 📘 **Aerial Bird vs Drone Classification & Detection**

>***This project aims to develop a deep learning-based solution that can classify aerial images into two categories — Bird or Drone — and optionally perform object detection to locate and label these objects in real-world scenes.
The solution will help in security surveillance, wildlife protection, and airspace safety where accurate identification between drones and birds is critical. The project involves building a Custom CNN classification model, leveraging transfer learning, and optionally implementing YOLOv8 for real-time object detection. The final solution will be deployed using Streamlit for interactive use.***

### **Deep Learning Project – Computer Vision (TensorFlow + YOLO + Streamlit)**

This project focuses on **classifying aerial objects** as **Bird** or **Drone** using:

* **Custom CNN**
* **Transfer Learning models** (ResNet50, EfficientNetB0)
* **Hybrid pipeline (optional)** – YOLOv8 detection + CNN classification
* **Streamlit Web App** for real-time inference

The project was built step-by-step from , covering dataset setup → model training → evaluation → UI deployment.

<!-- Top-line badges -->
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.20.0-orange.svg)]()
[![Model Accuracy](https://img.shields.io/badge/accuracy-81.86%25-brightgreen.svg)]()
[![F1-score](https://img.shields.io/badge/F1-0.802-blueviolet.svg)]()
[![Build](https://img.shields.io/badge/build-manual-lightgrey.svg)]()

---

# 🧭 **Table of Contents**

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Folder Structure](#folder-structure)
4. [Environment Setup](#environment-setup)
5. [Day-by-Day Workflow](#day-by-day-workflow)
6. [Model Architectures](#model-architectures)
7. [Training Results](#training-results)
8. [Evaluation (Confusion Matrix & Metrics)](#evaluation)
9. [Streamlit App](#streamlit-app)
10. [How to Run](#how-to-run)
11. [Future Improvements](#future-improvements)
12. [Acknowledgements](#acknowledgements)

---


# 📍 **Project Overview**

This project solves a common challenge in security, surveillance, and wildlife monitoring:

> **Identify whether the object in an aerial image is a Bird or a Drone.**

### Technologies used:

* **TensorFlow / Keras**
* **OpenCV**
* **NumPy, Pandas**
* **Matplotlib, Seaborn**
* **scikit-learn**
* **Albumentations**
* **ultralytics (YOLOv8)**
* **Streamlit**

---

# 📂 **Dataset Description**

* Total Images: **~3320+**
* Two classes:

  * **Bird**
  * **Drone**
* Split:

  * Train: **2662 images**
  * Validation: **442 images**
  * Test: **215 images**
* Balanced dataset (approx 53% birds, 47% drones)
* Images resized to **224 × 224**

Sample images:

---

# 🗂️ **Folder Structure**

```
Aerial Object Classification/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── classification_dataset/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│
├── models/
│   ├── tf_custom_cnn_best.h5
│   ├── tf_resnet50_best.h5
│   ├── tf_resnet50_finetuned_best.h5
│   ├── tf_efficientnetb0_best.h5
│   └── yolo_best.pt
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_data_augmentation.ipynb
│   ├── 04_custom_cnn.ipynb
│   ├── 05_transfer_learning.ipynb
│   └── 06_evaluation.ipynb
│
├── reports/
│   ├── confusion_matrix_custom_cnn.png
│   ├── training_curves_resnet50.png
│   └── comparison_table.png
│
└── src/
    ├── data_loader_tf.py
    ├── data_augmentation_tf.py
    ├── models/
    │   ├── custom_cnn_tf.py
    │   ├── transfer_learning_tf.py
    │   └── yolo_detector.py
```

---

# ⚙️ **Environment Setup**

## Create virtual environment

```
python -m venv venv
```

## Activate environment

Windows:

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

## Install requirements

```
pip install -r requirements.txt
```

---

# 🗓️ **Workflow**

## **EDA & Setup**

* Created project folder
* Created `venv`
* Installed TensorFlow, Torch, Albumentations, Streamlit, YOLO
* Loaded and visualized dataset
* Counted class distribution

## **Data Preprocessing**

* Wrote TensorFlow `tf.data` loader
* Normalized dataset to `[0, 1]`
* Verified pipeline shapes

## **Data Augmentation**

Using TensorFlow layers:

* Rotation
* Horizontal Flip
* Random Zoom
* Random Brightness
* Random Crop

Visualized augmented batches.

## **Custom CNN**

Architecture (13M params):

* 4 Conv blocks (Conv → BN → ReLU → MaxPool)
* Dense(256) → Dropout
* Dense(1, Sigmoid)

Performance:

* **Accuracy:** 81.86%
* **Precision:** 76.70%
* **Recall:** 84.04%

## **Transfer Learning**

Implemented:

* **ResNet50**
* **EfficientNetB0**
* Frozen base + custom head
* Fine-tuning top layers

Saved models:


**Models**
- tf_custom_cnn_best.h5 ![size](https://img.shields.io/badge/size-54MB-lightgrey)
- tf_resnet50_best.h5 ![size](https://img.shields.io/badge/size-98MB-lightgrey)
- tf_efficientnetb0_best.h5 ![size](https://img.shields.io/badge/size-29MB-lightgrey)



## **Evaluation**

* Confusion matrix
* Precision, Recall, F1-score
* Comparison of all models
* Plotted training curves

## **Streamlit App**

Built full web app:

* Image upload
* Auto model detection
* Preprocessing
* Prediction + probability
* UI improvements

---

# 🧠 **Model Architectures**

Here is a simplified block diagram for the **Custom CNN**:

```
Input (224,224,3)
      ↓
Conv (32) → BN → ReLU → MaxPool
      ↓
Conv (64) → BN → ReLU → MaxPool
      ↓
Conv (128) → BN → ReLU → MaxPool
      ↓
Conv (256) → BN → ReLU → MaxPool
      ↓
Flatten
Dense(256) + Dropout
Dense(1, Sigmoid)
```

---

# 📊 **Training Results**

Example (fill in with your actual metrics):

| Model                 | Accuracy | Precision | Recall | F1-score | Size |
| --------------------- | -------- | --------- | ------ | -------- | ---- |
| Custom CNN            | 0.8186   | 0.7670    | 0.8404 | 0.802    | 54MB |
| ResNet50 (frozen)     | 0.xxx    | 0.xxx     | 0.xxx  | 0.xxx    | 98MB |
| ResNet50 (fine-tuned) | 0.xxx    | 0.xxx     | 0.xxx  | 0.xxx    | 98MB |
| EfficientNetB0        | 0.xxx    | 0.xxx     | 0.xxx  | 0.xxx    | 29MB |

---

# 🔍 **Evaluation**

## 📌 Confusion Matrix (Custom CNN)

## 📌 Classification Report

* Accuracy
* Precision
* Recall
* F1-score
* Support

Generated using:

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

---

# 🌐 **Streamlit App**

Full app features:

* Auto-detects all `.h5` models from `/models`
* File uploader
* Image preprocessing
* Model prediction
* Confidence score visualization
* Clean UI with progress bar

Run with:

```
streamlit run app/streamlit_app.py
```

Example UI layout:

---

# ▶️ **How to Run the Project**

### 1. Clone project

```
git clone <repo_url>
```

### 2. Create & activate venv

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run training (optional)

Open Jupyter Notebook and run notebooks inside `/notebooks`.

### 5. Run Streamlit app

```
streamlit run app/streamlit_app.py
```

---

# 🚀 **Future Improvements**

* Add YOLOv8 object detection + crop → CNN classification pipeline
* Deploy on **Streamlit Cloud**, **Render**, or **Docker**
* Add **Grad-CAM heatmaps**
* Add **real-time webcam inference**
* Build **mobile app** using TFLite
* Add multi-class expansion (Bird species, Drone types)

---

# 🙏 **Acknowledgements**

* TensorFlow team
* Albumentations library
* Ultralytics YOLOv8
* Streamlit
* Open-source contributors
* Aerial dataset providers

---

