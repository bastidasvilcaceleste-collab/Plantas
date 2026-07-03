import os
import re
import shutil
from collections import defaultdict
from PIL import Image


DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'clean_final')

KNOWN_CROPS = sorted([
    'Apple', 'Banana', 'Bellpepper', 'Carrot', 'Cassava', 'Cherry', 'Chili',
    'Coffee', 'Corn', 'Cucumber', 'Gauva', 'Grape', 'Guava', 'Jamun', 'Jujube',
    'Lemon', 'Mango', 'Orange', 'Peach', 'Pepper', 'Pepper_bell', 'Pomegranate',
    'Potato', 'Rice', 'Soybean', 'Strawberry', 'Sugarcane', 'Tea', 'Tomato', 'Wheat',
], key=len, reverse=True)

CROP_SPELLING_FIXES = {
    'Gauva': 'Guava',
}

report = {
    'merged': [],
    'renamed': [],
    'corrupt_deleted': [],
    'images_moved': 0,
    'classes_before': 0,
    'classes_after': 0,
}


def split_camelcase(s):
    return re.sub(r'([a-z])([A-Z])', r'\1_\2', s)


def title_case_words(s):
    s = split_camelcase(s)
    return '_'.join(w.capitalize() for w in s.split('_') if w)


def normalize_dirname(dirname):
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', dirname.strip())

    if '___' in name or '__' in name:
        parts = name.replace('___', '|').replace('__', '|').split('|')
        crop_part = parts[0].strip('_')
        disease_part = '_'.join(p.strip('_') for p in parts[1:])
    else:
        crop_part = None
        for crop in KNOWN_CROPS:
            if not name.lower().startswith(crop.lower()):
                continue
            rest = name[len(crop):]
            if rest and rest[0] == '_':
                crop_part = crop
                disease_part = rest[1:]
                break
        if crop_part is None:
            return None

    disease_part = disease_part.replace('(', '').replace(')', '').replace('  ', ' ').strip()

    crop_clean = title_case_words(crop_part)

    lower_crop = crop_clean.lower()
    if disease_part.lower().startswith(lower_crop + '_'):
        disease_part = disease_part[len(crop_clean) + 1:]

    crop_clean = CROP_SPELLING_FIXES.get(crop_clean, crop_clean)

    disease_clean = title_case_words(disease_part)

    return f'{crop_clean}_{disease_clean}'


def is_image_valid(filepath):
    if os.path.getsize(filepath) == 0:
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f'ERROR: Dataset directory not found: {DATASET_DIR}')
        return

    raw_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    report['classes_before'] = len(raw_dirs)

    print(f'Dataset: {DATASET_DIR}')
    print(f'Classes detected (before): {len(raw_dirs)}')
    print('=' * 60)

    normalization_map = {}
    skipped = []
    for d in raw_dirs:
        norm = normalize_dirname(d)
        if norm is None:
            skipped.append(d)
        else:
            normalization_map[d] = norm

    if skipped:
        print(f'\n[!] {len(skipped)} directories could not be normalized (skipped):')
        for s in skipped:
            print(f'    {s}')

    groups = defaultdict(list)
    for orig, norm in normalization_map.items():
        groups[norm].append(orig)

    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    if duplicates:
        print(f'\n{" DUPLICATE CLASSES DETECTED ":=^60}')
        for norm, dirs in sorted(duplicates.items()):
            total_files = sum(
                len(os.listdir(os.path.join(DATASET_DIR, d)))
                for d in dirs if os.path.isdir(os.path.join(DATASET_DIR, d))
            )
            print(f'\n  {norm}  ({total_files} total files)')
            for i, d in enumerate(dirs):
                path = os.path.join(DATASET_DIR, d)
                count = len(os.listdir(path)) if os.path.isdir(path) else 0
                marker = ' [CANONICAL]' if i == 0 else ''
                print(f'    -> {d} ({count} files){marker}')
    else:
        print('\n  No duplicate classes found.')

    singles = {k: v for k, v in groups.items() if len(v) == 1}
    only_renamed = {k: v[0] for k, v in singles.items() if v[0] != k}
    if only_renamed:
        print(f'\n{" CLASSES TO RENAME ":=^60}')
        for norm, orig in sorted(only_renamed.items()):
            print(f'  {orig}  >>  {norm}')

    print('=' * 60)
    print('\nProcessing...\n')

    for norm, dirs in sorted(groups.items()):
        target = os.path.join(DATASET_DIR, norm)
        os.makedirs(target, exist_ok=True)

        for i, d in enumerate(dirs):
            source = os.path.join(DATASET_DIR, d)
            if not os.path.isdir(source):
                continue
            if os.path.abspath(source) == os.path.abspath(target):
                continue

            files = [
                f for f in os.listdir(source)
                if os.path.isfile(os.path.join(source, f))
            ]
            for f in files:
                src_file = os.path.join(source, f)
                dst_file = os.path.join(target, f)
                counter = 1
                base, ext = os.path.splitext(f)
                while os.path.exists(dst_file):
                    dst_file = os.path.join(target, f'{base}_{counter}{ext}')
                    counter += 1
                shutil.move(src_file, dst_file)
                report['images_moved'] += 1

            try:
                os.rmdir(source)
                print(f'  [+] Merged & removed: {d} -> {norm}  ({len(files)} files)')
            except OSError:
                remaining = os.listdir(source)
                if remaining:
                    print(f'  [!] Not empty after move: {d} (remaining: {remaining})')
                else:
                    os.rmdir(source)
                    print(f'  [+] Removed: {d}')

    removed_dirs = [d for dirs in duplicates.values() for d in dirs[1:]]
    removed_dirs += [d for norm, d in only_renamed.items()]
    for norm, dirs in groups.items():
        target = os.path.join(DATASET_DIR, norm)
        if not os.path.isdir(target):
            os.makedirs(target, exist_ok=True)

    print(f'\n{" CORRUPT IMAGE SCAN ":=^60}')
    all_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    corrupt_count = 0
    for d in all_dirs:
        dirpath = os.path.join(DATASET_DIR, d)
        for f in sorted(os.listdir(dirpath)):
            filepath = os.path.join(dirpath, f)
            if not os.path.isfile(filepath):
                continue
            if not is_image_valid(filepath):
                os.remove(filepath)
                report['corrupt_deleted'].append(filepath)
                corrupt_count += 1
                print(f'  [X] Deleted corrupt: {d}/{f}')

    if corrupt_count == 0:
        print('  No corrupt images found.')

    print(f'\n{" CLEANING EMPTY DIRS ":=^60}')
    empty_count = 0
    for d in sorted(os.listdir(DATASET_DIR)):
        dirpath = os.path.join(DATASET_DIR, d)
        if os.path.isdir(dirpath):
            try:
                os.rmdir(dirpath)
                print(f'  Removed empty: {d}')
                empty_count += 1
            except OSError:
                pass
    if empty_count == 0:
        print('  No empty directories.')

    final_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    report['classes_after'] = len(final_dirs)

    print(f'\n{"=" * 60}')
    print(f'{" FINAL REPORT ":=^60}')
    print(f'{"=" * 60}')
    print(f'  Classes before:          {report["classes_before"]}')
    print(f'  Classes after:           {report["classes_after"]}')
    print(f'  Duplicate groups merged: {len(duplicates)}')
    print(f'  Directories renamed:     {len(only_renamed)}')
    print(f'  Images moved:            {report["images_moved"]}')
    print(f'  Corrupt images deleted:  {corrupt_count}')
    print(f'  Empty dirs removed:      {empty_count}')
    print(f'  Classes removed:         {report["classes_before"] - report["classes_after"]}')
    print(f'{"=" * 60}')

    print(f'\n{" FINAL STRUCTURE ":=^60}')
    for d in final_dirs:
        dirpath = os.path.join(DATASET_DIR, d)
        count = len(os.listdir(dirpath))
        print(f'  {d}  ({count} images)')
    print(f'{"=" * 60}')
    print(f'  Total: {len(final_dirs)} classes')


if __name__ == '__main__':
    main()
