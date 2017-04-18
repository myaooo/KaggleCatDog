import os
import pickle
import random
import re

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2

DATA_ROOT = os.path.abspath(os.path.join(__file__, '../../../data'))
RAW_TRAIN_DIR = os.path.join(DATA_ROOT, 'train/')
RAW_TEST_DIR = os.path.join(DATA_ROOT, 'test/')
PREPROCESSED = os.path.join(DATA_ROOT, 'preprocessed/')
TRAIN_DIR = os.path.join(PREPROCESSED, 'train/')
TEST_DIR = os.path.join(PREPROCESSED, 'test/')

IMG_SIZE = (128, 128)
CHANNELS = 3
NUM_LABELS = 2
BATCH_SIZE = 25

SEED = None

data_type = np.float32
label_type = np.int8


def before_save(file_or_dir):
    """
    make sure that the dedicated path exists (create if not exist)
    :param file_or_dir:
    :return:
    """
    dir_name = os.path.dirname(os.path.abspath(file_or_dir))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return img


def read_images(file_paths):
    images = []
    for file_path in file_paths:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
        images.append(img)
    return images


def write_images(images, names):
    for i, image in enumerate(images):
        cv2.imwrite(names[i], image)


def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm


def preprocess_images(image_paths, target_size=IMG_SIZE):
    """
    Preprocess the raw data into the same sized images
    :param image_paths:
    :param target_size:
    :return:
    """
    print('*****Preprocessing*****')
    count = len(image_paths)
    images = []
    for i, image_path in enumerate(image_paths):
        if (i + 1) % 1000 == 0:
            print("Resizing {}/{}".format(i + 1, count))
        image = read_image(image_path)
        images.append(cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC))
    return images


def maybe_preprocess(train=True, ratio=None):
    if train:
        raw_dir = RAW_TRAIN_DIR
        target_dir = TRAIN_DIR
        dirs = os.listdir(raw_dir)
    else:
        raw_dir = RAW_TEST_DIR
        target_dir = TEST_DIR
        dirs = sort_by_name(os.listdir(raw_dir))
    print("Preprocessing {:s} data...".format('train' if train else 'test'))

    paths = [target_dir + i for i in dirs]
    labels = [0 if 'cat' in i else 1 if 'dog' in i else -1 for i in dirs]
    if os.path.exists(target_dir):
        print("Preprocessed {:s} data existed. Reading...".format('train' if train else 'test'))
        images = read_images(paths)
    else:
        before_save(paths[0])
        raw_paths = [raw_dir + i for i in dirs]
        images = preprocess_images(raw_paths)
        write_images(images, paths)
    if ratio is not None:
        cats = [images[i] for i, label in enumerate(labels) if label == 0]
        dogs = [images[i] for i, label in enumerate(labels) if label == 1]
        cat_train, cat_valid = split_data(cats, [1 - ratio, ratio])
        dog_train, dog_valid = split_data(dogs, [1 - ratio, ratio])
        images = cat_train + dog_train
        labels = [0] * len(cat_train) + [1] * len(dog_train)
        valid_images = cat_valid + dog_valid
        valid_labels = [0] * len(cat_valid) + [1] * len(dog_valid)
        images, labels = zip(*shuffle_data(list(zip(images, labels))))
        data, labels = format_data(list(images), list(labels))
        data2, labels2 = format_data(valid_images, valid_labels)
        return [(data, labels), (data2, labels2)]
    if train:
        images, labels = zip(*shuffle_data(list(zip(images, labels))))
        data, labels = format_data(list(images), list(labels))
        # labels = np.array(labels, dtype=label_type)
        return [(data, labels)]
    else:
        # print(dirs)
        return [format_data(images, labels)]


def sort_by_name(dirs, remove='.jpg'):
    """
    Remove the format string from dirs and sort by numerical order of the files
    :param dirs:
    :param remove:
    :return:
    """
    remove_len = len(remove)
    return sorted(dirs, key=lambda dir_: int(dir_[:-remove_len]))


def maybe_calculate(filename, cal_fn, *args, **kwargs):
    """
    Check whether a cached .pkl file exists.
    If exists, directly load the file and return,
    Else, call the `cal_fn`, dump the results to .pkl file specified by `filename`, and return the results.
    :param filename: the name of the target cached file
    :param cal_fn: a function that maybe called with `*args` and `**kwargs` if no cached file is found.
    :return: the pickle dumped object, if cache file exists, else return the return value of cal_fn
    """
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            results = pickle.loads(f.read())
    else:
        results = cal_fn(*args, **kwargs)
        before_save(filename)
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    return results


def prep_data(valid_ratio=0.2, test=False):
    assert 0 < valid_ratio < 1
    train_file = os.path.join(DATA_ROOT, 'tmp/train.pkl')
    train, valid = maybe_calculate(train_file, maybe_preprocess, True, valid_ratio)
    train = generate_data(train[0], train[1], BATCH_SIZE, True)
    valid = generate_data(valid[0], valid[1], BATCH_SIZE, False)
    test_data = None
    if test:
        test_file = os.path.join(DATA_ROOT, 'tmp/test.pkl')
        test_data, _ = maybe_calculate(test_file, maybe_preprocess, False)[0]

    return train, valid, test_data


def shuffle_data(data_list):
    random.seed(SEED)
    random.shuffle(data_list)
    return data_list


def split_data(data_list, ratios):
    assert isinstance(ratios, list)
    splitted = []
    start = 0
    length = len(data_list)
    for ratio in ratios:
        end = start + round(ratio * length)
        splitted.append(data_list[start:end])
        start = end
    return splitted


def format_data(images, labels=None):
    """
    Format a list of images to standard numpy.array of shape [n, img_size[0], img_size[1], channels]
    :param images: a list of images shaped [channels, rows, cols]
    :return: an instance of np.array
    """
    # _images = [image.T for image in images]
    data = np.stack(images, axis=0).astype(data_type) / 255 - 0.5
    if labels is not None:
        labels = np.array(labels, label_type)
    return data, labels


def generate_data(X, y, batch_size=32, train=True):
    """
    Using the returned data and label from maybe_preprocess / format_data,
    return a keras data generator.
    Only intended to use for training data
    :param X: a 4D array, formatted data
    :param y: a 1D array, label array
    :param batch_size:
    :param train
    :return: a keras generator
    """
    if train:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    else:
        datagen = ImageDataGenerator()
    data_generator = datagen.flow(
        X, y,
        batch_size=batch_size)
    return data_generator


if __name__ == '__main__':
    train, train_labels = maybe_preprocess()[0]
    test, test_labels = maybe_preprocess(False)[0]
