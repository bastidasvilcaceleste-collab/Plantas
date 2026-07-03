import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'datasets')
SIMPLE_DIR = os.path.join(BASE, 'data_simple')

TARGET_PREFIXES = ('Potato_', 'Tomato_', 'Corn_', 'Maize_')


def count_images(path):
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]) if os.path.isdir(path) else 0


def copy_classes(src_dir, dst_dir, split_name):
    if not os.path.isdir(src_dir):
        print(f'  SKIP {src_dir} (not found)')
        return 0
    total = 0
    for cls in sorted(os.listdir(src_dir)):
        if not cls.startswith(TARGET_PREFIXES):
            continue
        src_cls = os.path.join(src_dir, cls)
        if not os.path.isdir(src_cls):
            continue
        dst_cls = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)
        count = 0
        for fname in os.listdir(src_cls):
            src_file = os.path.join(src_cls, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(dst_cls, fname))
                count += 1
        total += count
        print(f'    {cls}: {count} images')
    print(f'  Total for {split_name}: {total} images')
    return total


def main():
    print('=== PREPARANDO DATASET SIMPLIFICADO (Papa, Tomate, Maiz) ===')
    if os.path.exists(SIMPLE_DIR):
        shutil.rmtree(SIMPLE_DIR)
        print(f'  Eliminado {SIMPLE_DIR}')

    for split in ['train', 'val', 'test']:
        src = os.path.join(DATA_DIR, split)
        dst = os.path.join(SIMPLE_DIR, split)
        os.makedirs(dst, exist_ok=True)
        print(f'\n{split.upper()}:')
        copy_classes(src, dst, split)

    print(f'\nResumen final:')
    for split in ['train', 'val', 'test']:
        path = os.path.join(SIMPLE_DIR, split)
        classes = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
        total_imgs = sum(count_images(os.path.join(path, c)) for c in classes)
        print(f'  {split}: {len(classes)} clases, {total_imgs} imagenes')

    print(f'\nDataset simplificado listo en: {SIMPLE_DIR}')


if __name__ == '__main__':
    main()
