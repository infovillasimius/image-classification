import os

import cv2
import numpy as np
from imutils import paths
from tensorflow.python.keras.preprocessing.image import img_to_array


def prepare_images(root_dir):
    image_paths = sorted(list(paths.list_images(root_dir)))
    image_set = os.path.basename(root_dir)
    print(image_set)
    data = []
    class_names = []
    labels = []

    for imagePath in image_paths:
        path = os.path.dirname(imagePath)
        class_name = os.path.basename(path)
        if class_name not in class_names:
            class_names.append(class_name)
        labels.append(class_names.index(class_name))
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        data.append(image)

    labels = np.array(labels)
    print('labels shape: ', labels.shape)
    data = np.array(data, dtype="float") / 255.0
    np.savez_compressed(image_set + '_images', data)
    np.savez_compressed(image_set + '_labels', labels)
    np.save('class_names', class_names)


# root_dir = 'data/train'
# prepare_images(root_dir)
root_dir = 'data/test'
prepare_images(root_dir)

