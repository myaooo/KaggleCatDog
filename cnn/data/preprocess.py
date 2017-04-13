import os
import pickle
import random

import numpy as np
import tensorflow as tf
import cv2

DATA_ROOT = os.path.abspath(os.path.join(__file__, '../../data'))
RAW_TRAIN_DIR = os.path.join(DATA_ROOT, 'train/')
RAW_TEST_DIR = os.path.join(DATA_ROOT, 'test/')
PREPROCESSED = os.path.join(DATA_ROOT, 'preprocessed/')
TRAIN_DIR = os.path.join(PREPROCESSED, 'train/')
TEST_DIR = os.path.join(PREPROCESSED, 'test/')

IMG_SIZE = (64, 64)
CHANNELS = 3
NUM_LABELS = 2

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
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
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
        if (i+1) % 1000 == 0:
            print("Resizing {}/{}".format(i+1, count))
        image = read_image(image_path)
        images.append(cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC))
    return images


def maybe_preprocess(train=True):
    if train:
        raw_dir = RAW_TRAIN_DIR
        target_dir = TRAIN_DIR
    else:
        raw_dir = RAW_TEST_DIR
        target_dir = TEST_DIR
    print("Preprocessing {:s} data...".format('train' if train else 'test'))
    dirs = os.listdir(raw_dir)
    paths = [target_dir + i for i in dirs]
    labels = [0 if 'cat' in i else 1 if 'dog' in i else -1 for i in dirs]
    if os.path.exists(target_dir):
        print("Preprocessed {:s} data existed. Reading...".format('train' if train else 'test'))
        images = read_images(paths)
    else:
        before_save(paths[0])
        raw_train_paths = [RAW_TRAIN_DIR + i for i in dirs]
        images = preprocess_images(raw_train_paths)
        # if train:
        #     cats = [images[i] for i, label in enumerate(labels) if label is 0]
        #     cat_paths = [os.path.join(target_dir, '../cat/') + dirs[i] for i, label in enumerate(labels) if label is 0]
        #     dogs = [images[i] for i, label in enumerate(labels) if label is 1]
        #     dog_paths = [os.path.join(target_dir, '../dog/') + dirs[i] for i, label in enumerate(labels) if label is 1]
        #     write_images(cats, cat_paths)
        #     write_images(dogs, dog_paths)
        write_images(images, paths)
    images, labels = zip(*shuffle_data(list(zip(images, labels))))
    data, labels = format_data(list(images), list(labels))
    # labels = np.array(labels, dtype=label_type)
    return data, labels


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
    train_data, train_labels = maybe_calculate(train_file, maybe_preprocess)
    train_data, valid_data = split_data(train_data, [1-valid_ratio, valid_ratio])
    train_labels, valid_labels = split_data(train_labels, [1-valid_ratio, valid_ratio])
    test_data = None
    if test:
        test_file = os.path.join(DATA_ROOT, 'tmp/test.pkl')
        test_data, _ = maybe_calculate(test_file, maybe_preprocess, False)
    return train_data, train_labels, valid_data, valid_labels, test_data, None


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
        end = start + round(ratio*length)
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
    data = np.stack(images, axis=0).astype(data_type)/255 - 0.5
    if labels is not None:
        labels = np.array(labels, label_type)
    return data, labels

# train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
# # train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
# # train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
#
# test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]


if __name__ == '__main__':
    train, train_labels = maybe_preprocess()