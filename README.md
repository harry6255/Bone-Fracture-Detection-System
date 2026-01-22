<p align="center">
  <img src="assets/banner.png" width="900"/>
</p>

<h1 align="center">🦴 Bone Fracture Detection System</h1>
<p align="center">
  <b>Recall-Optimized Deep Learning Framework for X-ray Image Analysis</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red">
  <img src="https://img.shields.io/badge/Medical%20AI-Research-blue">
  <img src="https://img.shields.io/badge/Streamlit-Web%20Demo-success">
  <img src="https://img.shields.io/badge/Status-Research--Ready-green">
</p>

---

## 📌 Overview
Bone fracture detection from X-ray images is a critical yet challenging task due to subtle fracture patterns, noise, and class imbalance. This repository presents a **recall-optimized deep learning system** designed to **minimize missed fracture cases**, which is essential for medical screening applications.

The project delivers an **end-to-end pipeline**:
- Data preprocessing
- Model training and evaluation
- Recall-focused optimization
- Real-time deployment via a web interface

---

## 🎯 Key Contributions
- ✔ Recall-optimized fracture detection using **ResNet-18**
- ✔ Transfer learning for limited medical datasets
- ✔ Class-weighted loss to reduce false negatives
- ✔ Clinically motivated evaluation metrics
- ✔ Deployable **Streamlit web application**
- ✔ Research-grade reproducibility

---

## 🧠 System Architecture
<p align="center">
  <img src="assets/architecture.png" width="750"/>
</p>

**Pipeline Explanation:**
1. X-ray image upload  
2. Image preprocessing (resize, normalization)  
3. Feature extraction using ResNet-18  
4. Softmax-based classification  
5. Prediction with confidence score  

---

## 📂 Dataset
- **Dataset Name:** FracAtlas  
- **Modality:** X-ray images  
- **Classes:** Fractured, Non-fractured  
- **Challenge:** Severe class imbalance  

> ⚠️ The dataset is not included in this repository.  
> Please download it separately and organize it as shown below.

---

## 📁 Project Structure
