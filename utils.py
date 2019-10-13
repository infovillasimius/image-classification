import os

import cv2
import numpy as np
from imutils import paths
from tensorflow.python.keras.preprocessing.image import img_to_array


def prova():
    image = cv2.imread('data/train/Bedroom/image_0001.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', image)
    cv2.imshow('Gray image', gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def prepare_dataset(root_dir):
    os.chdir('/home/anto/PycharmProjects/cv/SUN397')
    image_paths = sorted(list(paths.list_images(root_dir)))
    image_set = os.path.basename(root_dir)
    print(image_set)
    data = []
    class_names = []
    labels = []

    for imagePath in image_paths:
        path = os.path.dirname(imagePath)
        class_name = os.path.basename(path)
        image_file_name = os.path.basename(imagePath)
        if class_name not in class_names:
            class_names.append(class_name)
        labels.append(class_names.index(class_name))
        image = cv2.imread(imagePath)
        if image is not None:
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            if not os.path.exists(class_name):
                os.makedirs(class_name)
            file_name = class_name + '/' + image_file_name
            cv2.imwrite(file_name, image)


def count_labels(root_dir):
    os.chdir('/home/anto/PycharmProjects/cv/sun2')
    image_paths = sorted(list(paths.list_images(root_dir)))
    image_set = os.path.basename(root_dir)
    print(image_set)
    last_class = ""
    labels = []
    n = 0

    for imagePath in image_paths:
        path = os.path.dirname(imagePath)
        class_name = os.path.basename(path)
        image_file_name = os.path.basename(imagePath)

        if len(last_class) == 0:
            last_class = class_name

        if last_class != class_name:
            labels.append((last_class, n))
            n = 1
            last_class = class_name
        else:
            n = n + 1

        if n<91:
            image = cv2.imread(imagePath)
            if image is None:
                n=n-1
            else:
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                if not os.path.exists('training/'+class_name):
                    os.makedirs('training/'+class_name)
                file_name = 'training/' + class_name + '/' + image_file_name
                cv2.imwrite(file_name, image)
        elif n<101:
            image = cv2.imread(imagePath)
            if image is None:
                n=n-1
            else:
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                if not os.path.exists('validation/' + class_name):
                    os.makedirs('validation/'+class_name)
                file_name = 'validation/' + class_name + '/' + image_file_name
                cv2.imwrite(file_name, image)

    return labels


#root_dir = '/media/anto/TOSHIBA EXT/università/magUnica/CV/SUN/SUN397'
#prepare_dataset(root_dir)
'''root_dir='/media/anto/TOSHIBA EXT/università/magUnica/CV/SUN/SUN397'
num = count_labels(root_dir)
print(list(num))

n=9999
for i in num:
    if n>i[1]:
        n=i[1]
        classe=i[0]

print('Il numero minimo di immagini è ', n,' per la classe ',classe)
'''
# root_dir = 'data/train'
# prepare_images(root_dir)
root_dir = 'data/test'
prepare_images(root_dir)