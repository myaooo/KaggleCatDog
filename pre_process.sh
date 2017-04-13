#!/usr/bin/env bash

VALIDATION_DIRECTORY="./data/preprocessed/new_valid/"
TRAIN_DIRECTORY="./data/preprocessed/new_train/"
LABELS_FILE="./data/labels.txt"
OUTPUT_DIRECTORY="./data/preprocessed/new/"

echo "cat
dog" > "${LABELS_FILE}"

rm -r "${TRAIN_DIRECTORY}"
rm -r "${VALIDATION_DIRECTORY}"
rm -r "${OUTPUT_DIRECTORY}"
mkdir -p "${TRAIN_DIRECTORY}/dog"
mkdir -p "${TRAIN_DIRECTORY}/cat"
mkdir -p "${OUTPUT_DIRECTORY}"

for name in ./data/preprocessed/train/dog.*; do
  cp "$name" "./data/preprocessed/new_train/dog"
done

for name in ./data/preprocessed/train/cat.*; do
  cp "$name" "./data/preprocessed/new_train/cat"
done

while read LABEL; do
  VALIDATION_DIR_FOR_LABEL="${VALIDATION_DIRECTORY}${LABEL}"
  TRAIN_DIR_FOR_LABEL="${TRAIN_DIRECTORY}${LABEL}"

  # Move the first randomly selected 100 images to the validation set.
  mkdir -p "${VALIDATION_DIR_FOR_LABEL}"
  VALIDATION_IMAGES=$(ls -1 "${TRAIN_DIR_FOR_LABEL}" | shuf | head -2500)
  for IMAGE in ${VALIDATION_IMAGES}; do
    mv -f "${TRAIN_DIRECTORY}${LABEL}/${IMAGE}" "${VALIDATION_DIR_FOR_LABEL}"
  done
done < "${LABELS_FILE}"

BUILD_SCRIPT="./cnn/data/build_image_data.py"

python "${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}"

cd ./models/;
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz;
tar xzf inception-v3-2016-03-01.tar.gz
