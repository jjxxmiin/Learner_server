#-*- encoding: utf8 -*-
import cv2
import os
import numpy as np
import flask
import pickle
import json
import werkzeug
import requests
import JFace
from time import sleep
from pathlib import Path

encoding_folder_path = 'datasets/encoding_datasets/'
train_folder_path = 'datasets/train_datasets/'
infer_folder_path = 'datasets/infer_datasets/'

def save_json(data, data_path):
    with open(data_path, 'w') as f:
        json.dump(data, f)


def load_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    return data


def train(data_path, name):
    # get image
    img = cv2.imread(data_path)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    filename = os.path.split(data_path)[-1]
    time = filename.split('.')[0]
    encoding = JFace.get_face_encodings(img)[0]

    if len(encoding) == 0:
        return "얼굴이 존재하지 않습니다 ㅠㅠ"

    else:
        # make data
        data = {}
        data['time'] = time
        data['name'] = name
        data['encoding'] = encoding.tolist()

        # save data
        if not os.path.exists(encoding_folder_path):
            os.makedirs(encoding_folder_path)

        save_path = os.path.join(encoding_folder_path, f"{name}_{time}.json")
        save_json(data, save_path)

        return "얼굴 등록이 완료되었습니다."


def infer(data_path):

    known_face_encodings = []
    known_face_names = []

    for encoding_file_path in os.listdir(encoding_folder_path):
        data = load_json(os.path.join(encoding_folder_path, encoding_file_path))

        known_face_encodings.append(data['encoding'])
        known_face_names.append(data['name'])

    # get image
    img = cv2.imread(data_path)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = JFace.get_face_locations(img)
    face_encodings = JFace.get_face_encodings(img, face_locations)

    name = "Unknown"

    face_names = []

    for face_encoding in face_encodings:
        matches = JFace.compare_faces(known_face_encodings, face_encoding)
        
        face_distances = JFace.get_face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_names


app = flask.Flask(__name__)


@app.route('/train', methods = ['GET', 'POST'])
def handle_train():
    imagefile = flask.request.files['image']
    label = flask.request.form['label']

    filename = imagefile.filename

    # filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + filename)

    save_folder_path = os.path.join(train_folder_path, label)

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    save_file_path = os.path.join(train_folder_path, label, filename)
    imagefile.save(save_file_path)

    sleep(1)

    message = train(save_file_path, label)

    return message


@app.route('/infer', methods = ['GET', 'POST'])
def handle_infer():
    imagefile = flask.request.files['image']
    filename = imagefile.filename

    
    if not os.path.exists(infer_folder_path):
        os.makedirs(infer_folder_path)

    save_file_path = os.path.join(infer_folder_path, filename)
    imagefile.save(save_file_path)

    labels = infer(save_file_path)

    label = ' '.join(labels)

    return label


app.run(host="0.0.0.0", port=5000, debug=True)