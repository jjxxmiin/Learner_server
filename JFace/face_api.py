import os
import numpy as np
import dlib
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face

pose_predictor_5_point = dlib.shape_predictor('JFace/models/shape_predictor_5_face_landmarks.dat')
pose_predictor_68_point = dlib.shape_predictor('JFace/models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('JFace/models/dlib_face_recognition_resnet_model_v1.dat')
cnn_face_detector = dlib.cnn_face_detection_model_v1("JFace/models/mmod_human_face_detector.dat")
face_detector = dlib.get_frontal_face_detector()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)


def load_model(load_path=None, classify=True):
    if load_path is not None:
        resnet.state_dict(torch.load(load_path))

    resnet.classify = classify


def get_dataset(dataset_path, encoding_model="dlib"):
    known_face_encoding = []
    known_face_names = []

    classes = os.listdir(dataset_path)[:10]

    for class_name in tqdm(classes, total=len(classes)):
        for file_path in os.listdir(dataset_path / class_name):
            image = load_image_file(dataset_path / class_name / file_path)

            if encoding_model is "cnn":
                encoding = get_cnn_face_encodings(image)
            else:
                encoding = get_face_encodings(image)

            if not len(encoding) == 0:
                known_face_encoding.append(encoding[0])
                known_face_names.append(class_name)

    return known_face_encoding, known_face_names


def get_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Face locations
    """
    faces = get_rect_faces(img, number_of_times_to_upsample, model)

    if model == "cnn":
        return [get_face_box(rect_to_css(face.rect), img.shape) for face in faces]
    else:
        return [get_face_box(rect_to_css(face), img.shape) for face in faces]


def rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def get_face_box(css, image_shape):
    """
    Face box
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def load_image_file(file, mode='RGB'):
    """
    Load image
    """
    im = Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def get_rect_faces(img, number_of_times_to_upsample=1, model="hog"):
    """
    Face
    """

    face_detector = get_detector(model)
    faces = face_detector(img, number_of_times_to_upsample)

    return faces


def get_detector(model="hog"):
    """
    Detector : cnn, hog
    """

    if model == "cnn":
        return cnn_face_detector
    else:
        return face_detector


def get_face_randmarks(img, face_locations=None, model="large"):
    """
    Face randmark
    """

    if face_locations is None:
        face_locations = get_rect_faces(img)
    else:
        face_locations = [css_to_rect(face_location) for face_location in face_locations]

    if model == "large":
        pose_predictor = pose_predictor_68_point
    else:
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(img, face_location) for face_location in face_locations]


def get_face_encodings(img, known_face_locations=None, num_jitters=1, model="small"):
    """
    Face encoding
    """

    raw_landmarks = get_face_randmarks(img, known_face_locations, model)

    return [np.array(face_encoder.compute_face_descriptor(img, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def get_cnn_face_encodings(img, locations=False):
    """
    Face cnn encoding
    """
    encodings = []
    boxes, probs = mtcnn.detect(img)

    if boxes is not None:
        for box in boxes:
            face_image = extract_face(img, box, save_path="./test.jpg")

            img_embedding = resnet(face_image.unsqueeze(0))
            encodings.append(img_embedding[0].detach().numpy())

    if locations:
        return encodings, boxes
    else:
        return encodings


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(get_face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


def get_face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))

    # sum(abs(a)^[2])^[1/2]
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)