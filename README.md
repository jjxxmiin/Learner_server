# Learner_server

- Flask : 1.1.2
- Werkzeug : 0.16.1
- opencv 
- dlib
- facenet-pytorch

1. Install requirements

```sh
pip install flask cmake
pip install opencv-python dlib facenet-pytorch
```

2. Make model folder

```
mkdir JFace/models
```

2. Download model files

- [https://github.com/ageitgey/face_recognition_models/tree/master/face_recognition_models/models](https://github.com/ageitgey/face_recognition_models/tree/master/face_recognition_models/models)

3. Start app

```sh
python api.py
```

## App

- [Here](https://github.com/jjeamin/Learner)

## Face recognition

1. Face Detection (Dlib)
2. Face Randmark (Dlib, MTCNN)
3. Face Alignment (Dlib)
4. Face Crop (Dlib)
5. Face Identification (Dlib, InceptionResnetV1)

## Reference

- [https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
