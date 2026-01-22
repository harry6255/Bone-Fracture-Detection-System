🦴 Bone Fracture Detection System (Recall-Optimized Deep Learning)
Overview

This repository presents an end-to-end deep learning framework for automated bone fracture detection from X-ray images. The system is designed with a clinical safety focus, prioritizing high recall (sensitivity) to minimize missed fracture cases. The project spans data preprocessing, model training, evaluation, and productization via a user-friendly web interface.

This work is part of a research-oriented ML pipeline, culminating in a deployable application and a camera-ready academic manuscript.

Key Contributions

Recall-optimized fracture detection using ResNet-18 with transfer learning

Class-imbalance handling through weighted loss

Clinically motivated evaluation focusing on false negative reduction

Reproducible training and inference pipeline

Streamlit-based deployment for real-time use

Dataset

Name: FracAtlas Dataset

Domain: Bone X-ray images

Classes:

Normal (Non-fractured)

Fractured

Source: Publicly available benchmark dataset for fracture detection

Challenge: Highly imbalanced class distribution (fewer fracture samples)

Note: The dataset is not included in this repository due to licensing. Please download it separately and follow the directory structure below.


Bone-Fracture-Detection/
│
├── app.py                  # Streamlit web application
├── train.py                # Model training script
├── dataset.py              # Custom PyTorch Dataset class
├── best_model.pth          # Trained model weights
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── data/
│   ├── images/
│   │   ├── Fractured/
│   │   └── Non_fractured/
│   └── processed_data/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
└── results/
    ├── confusion_matrix.png
    └── classification_report.txt

Methodology
Model Architecture

Backbone: ResNet-18

Pretrained on: ImageNet

Modified fully connected layers for binary classification

Key Design Choices

Transfer Learning to address limited medical data

Class-weighted Cross Entropy Loss to penalize missed fractures

Recall-based model checkpointing

Lightweight architecture for deployment efficiency


| Class     | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| Normal    | 0.94      | 0.84   | 0.88     |
| Fractured | 0.49      | 0.75   | 0.60     |

Installation
Requirements

Python ≥ 3.8

PyTorch

torchvision

Streamlit

NumPy

Pandas

scikit-learn

Pillow

Matplotlib

Seaborn
pip install -r requirements.txt

Training the Model

To train the model from scratch:
python train.py

Running the Web Application (Demo):
Local Deployment
streamlit run app.py

Features

Upload X-ray image

Automatic preprocessing

Real-time fracture prediction

Confidence score display

Reproducibility

This project is designed to be fully reproducible:

Fixed preprocessing pipeline

Explicit model architecture

Documented hyperparameters

Public dataset

Trained weights provided

Research Context

This work supports an academic manuscript titled:

“Recall-Optimized Deep Learning Framework for Bone Fracture Detection in X-ray Images Using ResNet-18”

The repository accompanies:

Comparative evaluation

Confusion matrix analysis

Deployment demonstration

Academic references

Limitations

Binary classification only

Single-view X-ray images

No explainability (Grad-CAM) in current version

Future Work

Grad-CAM visual explanations

Multi-class fracture localization

Mobile application deployment

Clinical dataset validation

Integration into hospital PACS systems

Ethical Disclaimer

This system is intended only as a decision-support tool and must not replace professional medical judgment.

Citation

If you use this work, please cite:

Haris Maqsood,
Recall-Optimized Deep Learning Framework for Bone Fracture Detection,
2026.

Author

Haris Maqsood
Computer Science / Artificial Intelligence
GitHub: (add your profile link here)

License

This project is released for academic and research purposes only.
