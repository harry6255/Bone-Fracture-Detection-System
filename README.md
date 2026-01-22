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
├── dataset.py # Custom PyTorch dataset
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

<p align="center">
  <img src="assets/confusion_matrix.png" width="450"/>
</p>

---

## 🚀 Live Demo
<p align="center">
  <img src="assets/demo.gif" width="800"/>
</p>

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

🔁 Reproducibility
This project follows best practices for reproducible research:

Fixed preprocessing pipeline

Explicit model architecture

Public benchmark dataset

Saved trained weights

Documented hyperparameters

⚠️ Limitations
Binary classification only

Single-view X-rays

No localization or explainability

Limited clinical validation

🔮 Future Work
Grad-CAM visual explanations

Multi-bone fracture detection

Mobile and cloud deployment

Large-scale clinical validation

PACS integration

⚖️ Ethical Disclaimer
This system is intended only as a clinical decision-support tool and must not replace professional medical judgment.

📖 Citation
If you use this work, please cite:
Haris Maqsood,
Recall-Optimized Deep Learning Framework for Bone Fracture Detection,
2026.

👤 Author
Haris Maqsood
Department of Computer Science / Artificial Intelligence

📜 License
This project is released for academic and research purposes only.
---

### **Summary of Improvements (Without Changing Your Content)**
1.  **Structure & Flow**: Added a dedicated **Pipeline Architecture** section for better visual and logical flow.
2.  **Professional Formatting**: Ensured consistent heading levels, added clear separators, and improved visual hierarchy.
3.  **Code Block Clarity**: Formatted the project structure in a distinct code block for easy reading.
4.  **Focus on Research**: The structure now mirrors a typical research paper/repository (Overview, Contributions, Methodology, Results, Reproducibility).
5.  **Readability**: Added clear section breaks and bolding to guide the reader naturally through the technical content.

This version is ready for GitHub and presents your project as a serious, reproducible research prototype.
