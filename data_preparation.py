import os
import random
import shutil
from pathlib import Path

TRAIN_SPLIT = 0.8

data_path = Path("./data")

# 1. Retrieve all folders (classes) and files (samples)
classes = {}
for subdir, dirs, files in os.walk(data_path):
    # Check if dir not a data dir
    if os.path.dirname(subdir):
        classes[os.path.basename(subdir)] = files

# 2. Create train and test folders
train_path = data_path / "train"
if not os.path.exists(train_path):
    os.mkdir(train_path)

test_path = data_path / "test"
if not os.path.exists(test_path):
    os.mkdir(test_path)

# 3. Split data
for cls, samples in classes.items():
    class_train_path = train_path / cls
    if not os.path.exists(class_train_path):
        os.mkdir(class_train_path)

    class_test_path = test_path / cls
    if not os.path.exists(class_test_path):
        os.mkdir(class_test_path)


    random.shuffle(samples)
    train_amount = int(len(samples) * TRAIN_SPLIT)

    print(f"Class {cls}: {train_amount=}, {len(samples)-train_amount=}")

    # To train
    for sample in samples[:train_amount]:
        src = data_path / cls / sample
        dst = class_train_path / sample
        shutil.copyfile(src, dst)

    # To test
    for sample in samples[train_amount:]:
        src = data_path / cls / sample
        dst = class_test_path / sample
        shutil.copyfile(src, dst)
