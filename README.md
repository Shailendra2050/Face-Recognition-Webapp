
![Demo](banner.png)




A real-time AI Face Recognition Web Application built using Python, Flask, and OpenCV (LBPH algorithm).
It allows users to register faces, train the model automatically, and recognize faces through a web interface.


🚀 Features

* 🧑 User face registration system
* 📸 Image-based dataset creation
* 🤖 Automatic model training (LBPH Face Recognizer)
* 👁️ Real-time face recognition via API
* 🌐 Flask web application
* 🎨 Clean UI with HTML/CSS/JS
* ⚡ Background training (non-blocking)
* ☁️ Ready for Render deployment (cloud-friendly)

⸻

🧠 How It Works

1. User registers with name + face images
2. Images are stored in dataset folder
3. After enough images, model is trained automatically
4. trainer.yml is generated
5. Recognition API predicts the user from input image

⸻

🏗️ Tech Stack

* Python 🐍
* Flask 🌐
* OpenCV 👁️
* NumPy 🔢
* Pillow 🖼️
* HTML, CSS, JavaScript 🎨

⸻

📁 Project Structure


Face-Recognition-main/
│
├── app.py                  # Flask backend
├── train.py               # Model training script
├── requirements.txt
│
├── cascades/              # Haarcascade files
│
├── recognizer/
│   ├── trainer.yml
│   ├── labels.pickle
│
├── static/
│   ├── css/
│   ├── js/
│   ├── dataset/
│   └── assets/
│       └── banner.png     # ⭐ Add banner image here
│
├── templates/
│   ├── index.html
│   ├── register.html

⚙️ Installation

1. Clone Repository
     git clone https://github.com/your-username/face-recognition-webapp.git
cd face-recognition-webapp

2. Create Virtual Environment
  python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies
   pip install -r requirements.txt

▶️ Run Project

Train Model
  python train.py

Start Flask App
 python app.py  