# 🧠 Brain Tumor Detection Web App – MSc Dissertation

This project is a full-stack web application I developed as part of my MSc Data Science dissertation at the University of Greenwich. The application uses a trained convolutional neural network (CNN) to detect brain tumors from uploaded medical images (JPG or PNG format).

---

## 🔍 Features

- 🔐 User registration and login (Django-based)
- 📂 Upload system for medical images with file type validation
- 🧠 Deep learning model integration for tumor prediction
- 📱 Responsive UI using Bootstrap
- 🧹 Secure handling and renaming of uploaded files

---

## 💼 My Role

I designed and built this project from scratch as my solo final-year dissertation. I handled all aspects of development — from designing the UI and setting up Django models to training and integrating the AI model using TensorFlow.

---

## 🛠️ Technologies Used

- **Frontend:** HTML5, CSS3, Bootstrap  
- **Backend:** Python, Django  
- **AI Model:** TensorFlow, Keras  
- **Database:** SQLite  
- **Other:** OpenCV, NumPy, Django Forms

---

## 📁 Project Structure

/braintumordetection
├── brain_tumor/
│ ├── forms.py
│ ├── models.py
│ ├── views.py
│ ├── templates/
│ └── static/
├── media/ # Uploaded images
├── static/
├── templates/
├── manage.py
└── requirements.txt


---

## ⚙️ Setup Instructions

1. Clone this repo:
```bash
git clone https://github.com/Iqra-99/braintumordetection
cd braintumordetection
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Django server:

bash
Copy
Edit
python manage.py runserver
Open http://127.0.0.1:8000 in your browser

Register → Login → Upload an image → Receive prediction!

