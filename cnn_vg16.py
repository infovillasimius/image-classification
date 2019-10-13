import os
from pathlib import Path

# Helper libraries
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

cwd = os.getcwd()
if cwd.find('data') + 4 == len(cwd):
    os.chdir('..')

training_features_file = Path('training_features.npy')
validation_features_file = Path('validation_features.npy')
test_features_file = Path('test_features.npy')
training_labels_file = Path('training_labels.npy')
validation_labels_file = Path('validation_labels.npy')
test_labels_file = Path('test_labels.npz')
test_images_file = Path('test_images.npz')
train_images_file = Path('train_images.npz')
train_labels_file = Path('train_labels.npz')
training_set_file = Path('training_set.npy')
validation_set_file = Path('validation_set.npy')

if training_set_file.is_file() and train_labels_file.is_file() and validation_set_file.is_file() \
        and validation_labels_file.is_file():
    training_set = np.load(training_set_file)
    validation_set = np.load(validation_set_file)
    training_labels = np.load(training_labels_file)
    validation_labels = np.load(validation_labels_file)
    print('Sets loaded')

else:
    train_images, train_labels = np.load(train_images_file), np.load(train_labels_file)
    train_images = train_images['arr_0']
    train_labels = train_labels['arr_0']

    training_set, validation_set, training_labels, validation_labels = train_test_split(train_images, train_labels,
                                                                                        test_size=0.10)
    np.save(training_labels_file, training_labels)
    np.save(validation_labels_file, validation_labels)
    np.save(training_set_file, training_set)
    np.save(validation_set_file, validation_set)
    del train_images, train_labels

model_file = Path('my_model.h5')
if model_file.is_file():
    model = load_model('my_model.h5')
else:
    baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)), classes=15)

    for layer in baseModel.layers[:-4]:
        layer.trainable = False

    model = Sequential()
    model.add(baseModel)
    model.add(Flatten(name="flatten"))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))

    opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.summary()

# model.load_weights("weights-improvement-02-0.36.hdf5")

file_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(training_set, training_labels,
                    epochs=100, validation_data=(validation_set, validation_labels),
                    callbacks=callbacks_list,
                    batch_size=10)

test_loss, test_acc = model.evaluate(validation_set, validation_labels)

print('Test accuracy:', test_acc)

del training_set, training_labels, validation_labels, validation_set

test_images, test_labels = np.load(test_images_file), np.load(test_labels_file)
test_images = test_images['arr_0']
test_labels = test_labels['arr_0']

predictions = model.predict(test_images)

p = []
for i in range(predictions.shape[0]):
    p.append(np.argmax(predictions[i]))
predictions = np.array(p)
print(predictions.shape)

print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))

# model.save('my_model.h5')
