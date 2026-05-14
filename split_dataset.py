import os
import shutil
import random
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
SRC = os.path.join(BASE, 'clean_final')
TRAIN = os.path.join(BASE, 'train')
VAL = os.path.join(BASE, 'val')
TEST = os.path.join(BASE, 'test')
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def split_dataset():
    random.seed(SEED)
    classes = sorted([d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))])
    total_images = 0
    splits = {'train': 0, 'val': 0, 'test': 0}

    for clazz in classes:
        src_dir = os.path.join(SRC, clazz)
        images = sorted([
            f for f in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, f))
        ])
        random.shuffle(images)
        n = len(images)
        if n < 3:
            print(f'  [WARN] {clazz} only has {n} images (need >=3 for split)')
            train_end = max(1, n - 2)
            val_end = train_end + 1
        else:
            train_end = int(n * TRAIN_RATIO)
            val_end = train_end + int(n * VAL_RATIO)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            dst_dir = os.path.join(BASE, split_name, clazz)
            os.makedirs(dst_dir, exist_ok=True)
            for f in files:
                shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            splits[split_name] += len(files)
        total_images += n

    print(f'Dataset split complete:')
    print(f'  Train: {splits["train"]} ({100*splits["train"]/total_images:.1f}%)')
    print(f'  Val:   {splits["val"]} ({100*splits["val"]/total_images:.1f}%)')
    print(f'  Test:  {splits["test"]} ({100*splits["test"]/total_images:.1f}%)')
    print(f'  Total: {total_images}')
    print(f'  Classes: {len(classes)}')
    return classes, splits


if __name__ == '__main__':
    split_dataset()
