# 🦋 Butterfly Image Classification

A Deep Learning based Butterfly Species Classification system built using **TensorFlow/Keras** and **Flask**. This project allows users to upload butterfly images and accurately predict their species through an interactive web interface and REST API.

---

## 🚀 Features

- 🔍 Single Image Prediction
- 📦 Batch Image Prediction
- 📊 Top-K Prediction Results
- 🌐 Modern Flask Web Interface
- 📡 REST API Support
- 🔄 Reload Model Without Restarting Server
- 📱 Responsive UI

---

## 🛠️ Technologies Used

- Python 3.11
- TensorFlow / Keras
- Flask
- Flask-CORS
- NumPy
- Pillow (PIL)
- HTML5
- CSS3
- JavaScript

---

## 📁 Project Structure

```
Butterfly-Image-Classification/
│
├── app.py
├── Model.ipynb
├── index.html
├── requirements.txt
├── README.md
├── best_butterfly_model.keras
├── butterfly_model.h5
├── class_names.json
├── Training_set.csv
└── Testing_set.csv
```

---

## 📥 Installation

Clone the repository

```bash
git clone https://github.com/Dabojit/Butterfly-Image-Classification.git
```

Move to project folder

```bash
cd Butterfly-Image-Classification
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

Open

```
Model.ipynb
```

Run all notebook cells.

After training, place

```
best_butterfly_model.keras
```

or

```
butterfly_model.h5
```

inside the project folder.

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | /health | Check server status |
| GET | /info | Model information |
| POST | /predict | Predict single image |
| POST | /predict_batch | Predict multiple images |
| POST | /predict_base64 | Predict Base64 image |
| POST | /reload | Reload trained model |

---

## 🧠 Model

- Architecture: EfficientNetB0
- Framework: TensorFlow/Keras
- Input Size: 224 × 224
- Output: Butterfly Species

---

## 📸 Screenshots

You can add screenshots here after running the application.

```
screenshots/home.png
screenshots/result.png
```

---

## 📌 Future Improvements

- Deploy on Render
- Docker Support
- Mobile Friendly UI
- Confidence Graph
- Model Performance Dashboard

---

## 👨‍💻 Author

**Dabojit Saha**

---

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub.
