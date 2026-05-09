from flask import Flask, render_template, request, jsonify
import os
import cv2
import base64
import numpy as np
import pickle
from PIL import Image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_path = os.path.join(BASE_DIR, "cascades", "haarcascade_frontalface_alt2.xml")
trainer_path = os.path.join(BASE_DIR, "recognizer", "trainer.yml")
labels_path = os.path.join(BASE_DIR, "recognizer", "labels.pickle")
dataset_path = os.path.join(BASE_DIR, "static", "dataset")

face_cascade = cv2.CascadeClassifier(face_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(trainer_path):
    recognizer.read(trainer_path)

labels = {}

if os.path.exists(labels_path):
    with open(labels_path, 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/save_user', methods=['POST'])
def save_user():
    data = request.json
    name = data['name']
    image = data['image']

    user_folder = os.path.join(dataset_path, name)
    os.makedirs(user_folder, exist_ok=True)

    image_data = image.split(',')[1]
    image_bytes = base64.b64decode(image_data)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    count = len(os.listdir(user_folder))

    image_path = os.path.join(user_folder, f'{count}.jpg')

    cv2.imwrite(image_path, img)

    if count >= 9:
        os.system('python train.py')
        return jsonify({'message': 'Training Completed'})

    return jsonify({'message': f'Image {count+1} Saved'})


@app.route('/recognize', methods=['POST'])
def recognize():

    try:
        data = request.json

        if not data or 'image' not in data:
            return jsonify({'result': 'No Image'})

        image = data['image']

        if ',' not in image:
            return jsonify({'result': 'Invalid Image'})

        image_data = image.split(',')[1]

        image_bytes = base64.b64decode(image_data)

        nparr = np.frombuffer(image_bytes, np.uint8)

        if nparr.size == 0:
            return jsonify({'result': 'Empty Frame'})

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'result': 'Frame Error'})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        result = 'Unknown'

        for (x, y, w, h) in faces:

            roi_gray = gray[y:y+h, x:x+w]

            roi_gray = cv2.resize(roi_gray, (200, 200))

            id_, conf = recognizer.predict(roi_gray)

            print("Confidence:", conf)

            if 4 <= conf <= 85:
                result = labels.get(id_, 'Unknown')

        return jsonify({'result': result})

    except Exception as e:
        print(e)
        return jsonify({'result': 'Error'})

#for running the app in localhost   

if __name__ == '__main__':
  app.run(debug=True)

# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=False)