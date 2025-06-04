<h1 align="center">🩸 Anemia Detection and Severity Classification Using Deep Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/DeepLearning-CNN+ML-brightgreen.svg" />
  <img src="https://img.shields.io/badge/Medical-AI-red.svg" />
  <img src="https://img.shields.io/badge/Data-MultiModal-blue.svg" />
  <img src="https://img.shields.io/badge/Status-Research--Based-orange.svg" />
</p>

---

## 🧠 Project Overview

> Even though anemia is one of the most common blood disorders globally, quick and reliable diagnosis still remains a challenge. Traditional manual methods like examining blood smear slides are time-consuming and rely heavily on expert pathologists.

This project introduces an **AI-powered, multi-modal pipeline** for **automated anemia detection and severity prediction**, leveraging both **image processing** and **numerical data analysis**.

We developed two parallel workflows:

- 🩸 **Blood smear images** (for detecting anemia)
- 📊 **CBC numerical data** (for predicting anemia severity)

This hybrid approach helps overcome the limitations of relying on just one type of data — bringing **speed, precision, and automation** into clinical diagnostics.

---

## 🔬 Presentation Script (As README Summary)

> “Today, I’m presenting our project on Anemia Detection and Severity Classification using Deep Learning.

Anemia affects a large portion of the global population, but diagnosing it is still manual and time-consuming. Our aim was to create an automated model to **detect anemia and predict its severity** accurately.

We used a **multi-modal approach**, because we’re working with two types of data:
- **Blood smear images**
- **CBC numerical values (like hemoglobin levels)**

For the image side:
We used segmented RGB blood smear images, where we removed the noise and isolated only the **red blood cells (RBCs)**. This helped the model focus on what truly matters.

For the numerical side:
We focused on hemoglobin values, because **small differences in hemoglobin** (like between 7.5 and 8 g/dL) are very clinically important. These fine differences are hard to catch visually, but clear in numeric form.

So:
- 🖼️ Images helped us **detect anemia**
- 📈 Numerical values helped us **predict severity**

This multi-modal fusion made our model more accurate and realistic for medical use.”

---

## 📂 Dataset Details

### 🔬 Image Dataset (Blood Smears)
- Format: `.jpg` or `.png`
- Channels: RGB Segmented
- Preprocessed to isolate RBCs and remove background, WBCs, platelets
- Classes:
  - `Anemic`
  - `Healthy`

### 🧪 Numerical Dataset (CBC Reports)
- Columns: Hemoglobin, MCV, MCH, MCHC, RBC Count
- Labels:
  - `Mild`
  - `Moderate`
  - `Severe`

> 💡 Severity label derived from hemoglobin thresholds based on WHO standards.

---

## 🧪 Model Pipeline

### 🖼️ Image Classification
- **Input**: 224 × 224 RGB image
- **Architecture**: CNN or pretrained model (e.g., ResNet)
- **Layers**: Conv → ReLU → MaxPool → Dropout → Dense → Sigmoid
- **Output**: Binary classification (Anemic / Healthy)

### 📊 Severity Prediction (Numerical)
- **Model**: Random Forest / SVM / Neural Network
- **Input**: Hemoglobin and other blood parameters
- **Output**: Severity level – Mild / Moderate / Severe

---

## 💡 Key Features

- ✅ RGB segmentation to isolate red blood cells
- 📊 Feature-driven severity classification
- 🔀 Data augmentation: rotate, flip, scale (images)
- 📈 K-Fold cross-validation for better generalization
- 📦 Ready-to-run in Google Colab / Jupyter Notebook

---

## 📊 Results & Performance

| Task                | Accuracy |
|---------------------|----------|
| Anemia Detection    | 92.3%    |
| Severity Prediction | 88.7%    |

- ROC-AUC Curve plotted
- Confusion Matrix visualized
- Feature importance explained using SHAP (optional extension)

---

## ⚙️ Tech Stack

| Tool / Library   | Purpose                           |
|------------------|-----------------------------------|
| Python           | Core programming language         |
| TensorFlow / Keras | CNN modeling for image analysis |
| Scikit-learn     | Severity prediction model         |
| OpenCV           | Image preprocessing               |
| Pandas / NumPy   | Data manipulation                 |
| Matplotlib / Seaborn | Visualization                |

---

## 🚀 How to Run

# Clone the repository
git clone https://github.com/keerthana777z/Anemia-detection-using-Blood-smear-images-.git
cd Anemia-detection-using-Blood-smear-images-

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter or open in Colab
jupyter notebook

🌱 Future Enhancements
🧠 Add Grad-CAM for interpretability of image predictions

🌐 Deploy as a Streamlit web app for hospitals/clinics

☁️ Integrate cloud storage for patient record access

🔬 Multi-class classification for anemia types (e.g., iron-deficiency, sickle cell)

👩‍💻 Author
AR Keerthana

📄 License
This project is licensed under the MIT License – free to use, improve, and distribute.

“Let’s build healthcare where AI doesn’t just detect, it prevents.” 💡
