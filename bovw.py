import os
from pathlib import Path
import cv2
import numpy as np
from imutils import paths
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

N = 500
data_path = Path ('data')


def train_vocabulary(path):
    k_means_trainer = cv2.BOWKMeansTrainer(N)

    image_paths = sorted(list(paths.list_images(path)))

    for path_to_image in image_paths:
        img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        detector = cv2.KAZE_create()
        key_points, descriptors = detector.detectAndCompute(img, None)

        if len(key_points) <= 0:
            continue

        descriptors = np.float32(descriptors)
        k_means_trainer.add(descriptors)
    vocabulary = k_means_trainer.cluster()

    return vocabulary


def get_visual_word_histogram_of_images(vocabulary, dataset):
    detector = cv2.KAZE_create()
    bow_ext = cv2.BOWImgDescriptorExtractor(detector, cv2.BFMatcher(cv2.NORM_L2))
    bow_ext.setVocabulary(vocabulary)
    image_paths = sorted(list(paths.list_images(dataset)))
    class_names = []
    labels = []
    bow_list = []

    for path_to_image in image_paths:
        path = os.path.dirname(path_to_image)
        class_name = os.path.basename(path)
        if class_name not in class_names:
            class_names.append(class_name)
        labels.append(class_names.index(class_name))
        image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        kp, des = detector.detectAndCompute(image, None)

        if bow_ext.compute(image, kp) is None:
            histogram = [0] * N
        else:
            histogram = bow_ext.compute(image, kp)[0]
        bow_list.append((histogram, path_to_image))
    return bow_list, labels, class_names


X_file = Path('bow_features.npy')
y_file = Path('bow_labels.npy')
class_names_file = Path('bow_classes.npy')

if X_file.is_file() and y_file.is_file() and class_names_file.is_file():
    X = np.load(X_file, allow_pickle=True)
    y = np.load(y_file, allow_pickle=True)
    class_names = np.load(class_names_file, allow_pickle=True)

else:
    vocabulary = train_vocabulary(data_path)
    bow_list, labels, class_names = get_visual_word_histogram_of_images(vocabulary, data_path)

    features = []
    for i in bow_list:
        features.append(i[0])
    X = np.array(features)
    y = np.array(labels)

    np.save('bow_features', X)
    np.save('bow_labels', y)
    np.save('bow_classes', class_names)

n = np.shape(X)[0]
print(n)
X = np.concatenate(X).ravel()
X = np.reshape(X, (n, N))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
classifier = svm.LinearSVC()
classifier.fit(X_train, y_train)

validation_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, validation_pred))
print(classification_report(y_test, validation_pred))
