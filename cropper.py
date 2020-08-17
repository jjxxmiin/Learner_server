import os
import cv2
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

show = False

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

origin_dataset_path = Path("train_origin_datasets")
crop_dataset_path = Path("train_crop_datasets")

if not os.path.exists(crop_dataset_path):
    os.mkdir(crop_dataset_path)

for class_path in origin_dataset_path.iterdir():
    class_path_parts = class_path.parts
    crop_class_path = crop_dataset_path / class_path_parts[-1]

    if not os.path.exists(crop_class_path):
        os.mkdir(crop_class_path)

    for file_path in class_path.iterdir():
        file_path_parts = file_path.parts
        crop_path = crop_class_path / file_path_parts[-1]

        if not crop_path.exists():
            img = Image.open(file_path)
            img = img.resize((512, 512))
            crop_img = mtcnn(img, save_path=str(crop_path))
            numpy_crop_img = crop_img.detach().numpy().transpose(1, 2, 0)

            print(str(crop_path))

            if show is True:
                cv2.imshow("crop", cv2.cvtColor(numpy_crop_img, cv2.COLOR_BGR2RGB))
                cv2.waitKey(10)

# img_embedding = resnet(img_cropped.unsqueeze(0))
#
# print(img_embedding)
#
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))
#
# print(img_probs)
# print(np.argmax(img_probs.detach().numpy()))


