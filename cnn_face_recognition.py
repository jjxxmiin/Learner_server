import cv2
import JFace
import numpy as np
from pathlib import Path

video_capture = cv2.VideoCapture(0)

# load_path = "./checkpoint/best_model.pth"
dataset_path = Path("train_origin_datasets")

JFace.load_model(load_path=None, classify=False)

known_face_encodings, known_face_names = JFace.get_dataset(dataset_path, encoding_model="cnn")

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_encodings, face_locations = JFace.get_cnn_face_encodings(rgb_frame, locations=True)

        face_names = []
        for face_encoding in face_encodings:
            matches = JFace.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = JFace.get_face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    if face_locations is not None:
        for (left, top, right, bottom), name in zip(face_locations, face_names):
            left, top, right, bottom = int(left), int(top), int(right), int(bottom) # x1, y2, x2, y1

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()