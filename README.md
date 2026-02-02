# Bone Fracture Detection System

Automated bone fracture detection from X‑ray images using a deep learning model.

## 🚀 Features

- 🔍 **Deep Learning Inference**: Predict fractures from bone X‑ray images.
- ⚙️ **Streamlit Interface**: Upload X‑ray images via a simple web interface.
- 📦 **Training Code Included**: Scripts/notebooks for training your own model.
- 🧠 **Easy Deployment**: Run locally or deploy on Streamlit Cloud.

## 📦 Prerequisites

- Python 3.8+
- `pip`
- Streamlit (`pip install streamlit`)
- PyTorch and other dependencies (from `requirements.txt`)

## 🛠 Installation

```bash
git clone https://github.com/harry6255/Bone-Fracture-Detection-System.git
cd Bone-Fracture-Detection-System
pip install -r requirements.txt
🔎 Usage
Run the Streamlit app locally:

streamlit run app.py
This will open a browser window where you can upload an X-ray image and get predictions.

🌐 Live Demo
Once deployed to Streamlit Cloud, replace the link below with your app’s URL:


⚠️ Note: Local testing works via streamlit run app.py.
The live demo link only works after deploying to Streamlit Cloud.

🧪 Training Your Own Model
Prepare a labeled dataset (fracture / normal).

Use scripts in traning_code/ for preprocessing and training.

Save the best model as best_fracture_model.pth.

🧑‍💻 Contributing
Fork the repository

Create a feature branch

Commit and push changes

Open a Pull Request
