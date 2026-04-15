# 🩺 Pneumonia Detection using Deep Learning

A deep learning-powered web application that detects **Pneumonia from Chest X-ray images** using an EfficientNet-based model. Built with a clean UI using Streamlit for real-time predictions.

---

## 🚀 Demo

> Upload a chest X-ray image and instantly get:
- Prediction (Normal / Pneumonia)
- Confidence score
- Visual feedback

---

## 🧠 Model Details

- **Architecture:** EfficientNet-B0  
- **Framework:** PyTorch  
- **Output:** Binary Classification (Normal vs Pneumonia)  
- **Activation:** Sigmoid  

The model is trained to identify lung opacity patterns and abnormalities associated with pneumonia.

---

## 📂 Project Structure

```
Pneumonia-Predictor/
│
├── backend/
│   └── pneumonia_efficientnet.pth   # (not included in repo)
│
├── app.py                           # Streamlit app
├── requirements.txt                 # Dependencies
├── notebook.ipynb                   # Training / experimentation
└── README.md
```

---

## ⚠️ Model & Dataset Download

The trained model and dataset are not included due to size limitations.

### 📦 Model
👉 Download from:  
https://drive.google.com/drive/folders/1o2ybVyZuvv6klrhGpxmGZE6NGTlQE_-J?usp=drive_link  

Place it inside:
```
backend/
```

### 📊 Dataset
👉 Download from:  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  

After extracting, place it inside:
```
xray/
```

---

## ⚙️ Installation

### 1. Clone the repository
```
git clone https://github.com/Dharun-2007/Pneumonia-Predictor.git
cd Pneumonia-Predictor
```

### 2. Create virtual environment (recommended)
```
conda create -p ./venv python=3.10
conda activate ./venv
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run app.py
```

---

## 🖼️ Features

- 📤 Upload chest X-ray images  
- 🤖 AI-based prediction  
- 📊 Confidence visualization  
- ⚡ Fast inference (CPU compatible)  
- 🧩 Clean and responsive UI  

---

## 🧩 Tech Stack

- Python  
- PyTorch  
- Streamlit  
- Torchvision  
- Matplotlib  

---

## 🚧 Limitations

- Model performance depends on dataset quality  
- Not a substitute for medical diagnosis  
- Requires manual model download  

---

## 🔮 Future Improvements

- Add Grad-CAM visualization  
- Deploy on cloud (Streamlit Cloud / Hugging Face Spaces)  
- Improve model accuracy with larger dataset  
- Add multi-class classification  

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is for educational purposes.

---

## 🙌 Acknowledgements

- Chest X-ray dataset from kaggle 
- PyTorch & Streamlit communities  

---

## 💡 Author

**Dharun**  
Aspiring Machine Learning Engineer 🚀
