import preprocess_data
import hyperparams
from tensorflow import keras
import tensorflow
import numpy as np

import os
"""
In order to use the same model, which was pre-trained on ImageNet, I made the input RGB. We cannot change this.

"""


#images_dir = "./Train_data - MRNET/"
#images_dir = "./Train_data - Pneumonia/"
# images_dir = "./Train_data - ABIDE"
# images_dir = "./Train_data - Skin Cancer"
images_dir = "./Train_data - Cataracts"


img_width = preprocess_data.img_width
img_height = preprocess_data.img_height
color_mode = preprocess_data.grayscale
batchsize = hyperparams.batch_size

train = keras.utils.image_dataset_from_directory(
    directory=images_dir,
    labels='inferred',
    color_mode=color_mode,
    label_mode='categorical',
    batch_size=batchsize,
    image_size=(img_width, img_height))


# Extracting Features and Labels
x_train = []
y_train = []

for feature, label in train:
    x_train.append(feature.numpy())
    y_train.append(label.numpy())


# Concatenate the lists to get the full 'x' and 'y' arrays
x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# get a list of true/false to identify labels
benign_indices = []
malignant_indices = []

for i, item in enumerate(y_train):
    if (item == np.array([0,1])).all():
        malignant_indices.append(i)
    else:
        benign_indices.append(i)
    continue


x_train_benign = np.array([x_train[i] for i in benign_indices])
y_train_benign = np.array([y_train[i] for i in benign_indices])

x_train_malignant = np.array([x_train[i] for i in malignant_indices])
y_train_malignant = np.array([y_train[i] for i in malignant_indices])

""" Downsample by selecting first 1341 images in malignant dataset"""
x_train_malignant_downsampled = x_train_malignant[:len(x_train_benign)]
y_train_malignant_downsampled = y_train_malignant[:len(y_train_benign)]

x_train_balanced = np.concatenate((x_train_benign, x_train_malignant_downsampled))
y_train_balanced = np.concatenate((y_train_benign, y_train_malignant_downsampled))

# print(x_train_balanced.shape)
# print(y_train_balanced.shape)

# x_train_balanced = np.concatenate(x_train_balanced, axis=0)
# y_train_balanced = np.concatenate(y_train_balanced, axis=0)


def preprocessLabels(label_array):
    labels = []
    for ele in label_array:
        if list(ele) == [0.0, 1.0]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)

y_train_labels_balanced = preprocessLabels(y_train_balanced)
y_train_labels = preprocessLabels(y_train)


if __name__ == "__main__":
    import db_helpers
    dataset = "Skin Cancer"
    sample_size = len(x_train)
    num_control = len(benign_indices)
    num_treatment = len(malignant_indices)
    db_helpers.populate_datasets(dataset, sample_size, num_control, num_treatment)



