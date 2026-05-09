import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR, "static", "dataset")
trainer_path = os.path.join(BASE_DIR, "recognizer", "trainer.yml")
labels_path = os.path.join(BASE_DIR, "recognizer", "labels.pickle")

face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, 'cascades', 'haarcascade_frontalface_alt2.xml')
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):

            path = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-').lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            pil_image = Image.open(path).convert('L')
            image_array = np.array(pil_image, 'uint8')

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]

                roi = cv2.resize(roi, (200, 200))

                x_train.append(roi)
                y_labels.append(id_)

with open(labels_path, 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save(trainer_path)

print('Training Completed Successfully')