import preprocess_data
import hyperparams
import keras
import tensorflow
import numpy as np
import preprocess_data

"""
In order to use the same model, which was pre-trained on ImageNet, I made the input RGB. We cannot change this.

"""

img_width = preprocess_data.img_width
img_height = preprocess_data.img_height
color_mode = preprocess_data.grayscale
batchsize = hyperparams.batch_size

train = keras.utils.image_dataset_from_directory(
    directory='./kaggle/input/skin_cancer_data/skin_cancer/train',
    labels='inferred',
    color_mode=color_mode,
    label_mode='categorical',
    batch_size=batchsize,
    image_size=(img_width, img_height))

validation = keras.utils.image_dataset_from_directory(
    directory='./kaggle/input/skin_cancer_data/skin_cancer/val',
    labels='inferred',
    color_mode=color_mode,
    label_mode='categorical',
    batch_size=batchsize,
    image_size=(img_width, img_height))

test = keras.utils.image_dataset_from_directory(
    directory='./kaggle/input/skin_cancer_data/skin_cancer/test',
    labels='inferred',
    color_mode=color_mode,
    label_mode='categorical',
    batch_size=batchsize,
    image_size=(img_width, img_height))

# Extracting Features and Labels
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature.numpy())
    y_train.append(label.numpy())

for feature, label in test:
    x_test.append(feature.numpy())
    y_test.append(label.numpy())

for feature, label in validation:
    x_val.append(feature.numpy())
    y_val.append(label.numpy())


# Concatenate the lists to get the full 'x' and 'y' arrays
x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# get a list of true/false to identify labels
normal_indices = []
skin_cancer_indices = []

for i, item in enumerate(y_train):
    if (item == np.array([0,1])).all():
        skin_cancer_indices.append(i)
    else:
        normal_indices.append(i)
    continue


x_train_normal = np.array([x_train[i] for i in normal_indices])
y_train_normal = np.array([y_train[i] for i in normal_indices])

x_train_skin_cancer = np.array([x_train[i] for i in skin_cancer_indices])
y_train_skin_cancer = np.array([y_train[i] for i in skin_cancer_indices])

""" Downsample by selecting first 1341 images in skin_cancer dataset"""
x_train_skin_cancer_downsampled = x_train_skin_cancer[:len(x_train_normal)]
y_train_skin_cancer_downsampled = y_train_skin_cancer[:len(y_train_normal)]

x_train_balanced = np.concatenate((x_train_normal, x_train_skin_cancer_downsampled))
y_train_balanced = np.concatenate((y_train_normal, y_train_skin_cancer_downsampled))

print(x_train_balanced.shape)
print(y_train_balanced.shape)

# x_train_balanced = np.concatenate(x_train_balanced, axis=0)
# y_train_balanced = np.concatenate(y_train_balanced, axis=0)

x_val = np.concatenate(x_val, axis=0)
y_val = np.concatenate(y_val, axis=0)

x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

def preprocessLabels(label_array):
    labels = []
    for ele in label_array:
        if list(ele) == [0.0, 1.0]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)

y_train_labels_balanced = preprocessLabels(y_train_balanced)
y_test_labels = preprocessLabels(y_test)


if __name__ == "__main__":
    import txtlogs
    txtlogs.logText("Dataset imported... details:")
    txtlogs.logText("-----------------------------------------------------")
    txtlogs.logText(f"Shape of 'x_train': {x_train.shape}")
    txtlogs.logText(f"Shape of 'y_train': {y_train.shape}")
    txtlogs.logText(f"Shape of 'x_train_balanced': {x_train_balanced.shape}")
    txtlogs.logText(f"Shape of 'y_train_balanced': {y_train_balanced.shape}")
    txtlogs.logText(f"Shape of 'x_test': {x_test.shape}")
    txtlogs.logText(f"Shape of 'y_test': {y_test.shape}")
    txtlogs.logText(f"Shape of 'x_val': {x_val.shape}")
    txtlogs.logText(f"Shape of 'y_val': {y_val.shape}")
    txtlogs.logText("-----------------------------------------------------")



