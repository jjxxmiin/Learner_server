import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("./train_origin_datasets/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")

# print(np.array(img))
#
# x1, y1, x2, y2 = 71, 59, 170, 190
#
# img = np.array(img)
#
# plt.imshow(img[y1: y2, x1: x2, :])
# plt.show()

from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face
mtcnn = MTCNN(keep_all=True)
boxes, probs, points = mtcnn.detect(img, landmarks=True)
# Draw boxes and save faces
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)

for i, (box, point) in enumerate(zip(boxes, points)):
    draw.rectangle(box.tolist(), width=5)
    for p in point:
        draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
    extract_face(img, box, save_path='detected_face_{}.png'.format(i))

img_draw.save('annotated_faces.png')