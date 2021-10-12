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

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def read_image(file_path) :
    stream = open( file_path.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)

    return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)


def train(file_path, name, mode):
    # get image
    img = read_image(file_path)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mode == 'gallery':
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    filename = os.path.split(file_path)[-1]
    time = filename.split('.')[0]

    try:
        encoding = JFace.get_face_encodings(img)[0]
    except:
        return "얼굴이 아닙니다. 사진을 다시 입력해주세요."

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


def infer(file_path):
    known_face_encodings = []
    known_face_names = []

    for encoding_file_path in os.listdir(encoding_folder_path):
        data = load_json(os.path.join(encoding_folder_path, encoding_file_path))

        known_face_encodings.append(data['encoding'])
        known_face_names.append(data['name'])

    # get image
    img = read_image(file_path)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = JFace.get_face_locations(img)
    face_encodings = JFace.get_face_encodings(img, face_locations)

    name = "누군지 모르겠습니다."

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


@app.route('/train/capture', methods = ['GET', 'POST'])
def handle_train_capture():
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

    print(save_file_path)
    message = train(save_file_path, label, mode='capture')

    return message


@app.route('/train/gallery', methods = ['GET', 'POST'])
def handle_train_gallery():
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

    print(save_file_path)
    message = train(save_file_path, label, mode='gallery')

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