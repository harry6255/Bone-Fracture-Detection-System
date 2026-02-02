
<h1 align="center">🦴 Bone Fracture Detection System</h1>
<p align="center">
  <b>Recall-Optimized Deep Learning Framework for X-ray Image Analysis</b>
</p>

<p align="center">
  <a href="https://harry6255-bone-fracture-detection-system-app-jskzgr.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Streamlit-Live%20Demo-success" alt="Live Demo">
  </a>
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

## 🔬 Pipeline Architecture
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
Bone-Fracture-Detection/
│
├── app.py # Streamlit application
├── train.py # Model training script
├── Dataset 
├── best_model.pth # Trained model weights
├── requirements.txt
├── README.md



---

## ⚙️ Methodology

### Model Architecture
- Backbone: **ResNet-18**
- Pretrained on ImageNet
- Custom fully connected classifier for binary classification

### Why Recall Optimization?
- False negatives are dangerous in healthcare
- Screening systems must prioritize sensitivity
- Model selection is based on **fracture recall**, not accuracy

---

## 📊 Results

### Quantitative Performance
| Class       | Precision | Recall | F1-score |
|------------|----------|--------|----------|
| Normal     | 0.94     | 0.84   | 0.88     |
| Fractured  | 0.49     | 0.75   | 0.60     |

- **Overall Accuracy:** 82%
- **Fracture Recall (Sensitivity):** 75%

---


### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Features:

Upload X-ray image

Automatic preprocessing

Real-time prediction

Confidence score visualization

---

⚠️ Limitations
Binary classification only

Single-view X-rays

No localization or explainability

Limited clinical validation
---
👤 Author
Haris Maqsood
Department of Computer Science / Artificial Intelligence

📜 License
This project is released for academic and research purposes only.
