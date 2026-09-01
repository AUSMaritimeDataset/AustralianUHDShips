from pathlib import Path
import yaml
from tqdm import tqdm
from numpy.random import default_rng, Generator
from typing import Iterable

TRAIN, VAL, TEST = "train", "val", "test"
SPLITS = (TRAIN, VAL, TEST)
MANIFEST = "manifest.yaml"

MANIFEST_CLASSES = "names"
MANIFEST_PATH = "path"

NDIGITS = 5

IMAGE_FOLDER = "images"
LABEL_FOLDER = "labels"
LABEL_SUFFIX = ".txt"

def write_split(
        images: list[Path], 
        labels: list[list[tuple[int, float, float, float, float]]], 
        root: Path, 
        split_name: str
):
    assert len(images) == len(labels)
    padding_size = len(str(len(images)))

    # set up directories
    image_folder = root / IMAGE_FOLDER / split_name
    image_folder.mkdir(parents=True)
    label_folder = root / LABEL_FOLDER / split_name
    label_folder.mkdir(parents=True)

    split_manifest = []
    print(f"Writing {split_name}...")
    for i, entry in tqdm(enumerate(zip(images, labels))):
        img, labels = entry
        new_name = (str(i).rjust(padding_size, "0") + img.suffix)

        # Write image
        img.copy(image_folder / new_name)

        # Write labels
        with open(label_folder / new_name.replace(img.suffix, LABEL_SUFFIX), 'w+', encoding='utf-8') as f:
            for row in labels:
                c, x ,y, w, h = row
                c = str(round(c))
                x = str(round(x, NDIGITS)).ljust(NDIGITS+2, "0") # Account for 0. at front
                y = str(round(y, NDIGITS)).ljust(NDIGITS+2, "0")
                w = str(round(w, NDIGITS)).ljust(NDIGITS+2, "0")
                h = str(round(h, NDIGITS)).ljust(NDIGITS+2, "0")
                f.write(" ".join((c,x,y,w,h)) + '\n')

        # Register entry
        split_manifest.append(f"./{IMAGE_FOLDER}/{split_name}/" + new_name)

    # Write out split manifest
    with open(root / f"{split_name}{LABEL_SUFFIX}", "w+", encoding='utf-8') as f:
        f.writelines([line + '\n' for line in split_manifest])

def write_yolo(
        images: dict[str, list[Path]], # Train, Val, Test
        labels: dict[str, list[list[tuple[int, float, float, float, float]]]],
        output_root: Path,
        class_map: dict[int, str],
):
    # Write splits
    for split in images:
        img, lab = images[split], labels[split]
        write_split(img, lab, output_root, split)

    # Write manifest
    manifest = {
        MANIFEST_CLASSES: class_map,
        MANIFEST_PATH: str(output_root)
    } | {split: f"{split}{LABEL_SUFFIX}" for split in images}
    with open(output_root / MANIFEST, 'w+', encoding='utf-8') as f:
        yaml.dump(manifest, f)


def random_split(inds: list[int], num: int, rng: Generator) -> tuple[list[int], list[int]]:
    sample = rng.choice(len(inds), size=num, replace=False, shuffle=False)
    sample.sort()
    replaced = []
    extracted = []
    for i,j in enumerate(inds):
        if i in sample:
            extracted.append(j)
        else:
            replaced.append(j)
    return replaced, extracted
    # TODO docstrings
def train_val_test_split(
        length: int, 
        val_prop: float = 0.15, 
        test_prop: float = 0.15, 
        seed: int = 0
) -> tuple[list[int], list[int], list[int]]: # Train, val, test    
    # Compute parameters
    rng = default_rng(seed)
    train_prop = 1 - (val_prop + test_prop)
    if train_prop < 0:
        raise ValueError("Proportions cannot sum to be greater than 1!")
    num_val = int(length * val_prop)
    num_test = int(length * test_prop)

    # Extract indexes
    train_indexes = list(range(length))
    train_indexes, val_indexes = random_split(train_indexes, num_val, rng)
    train_indexes, test_indexes = random_split(train_indexes, num_test, rng)

    return train_indexes, val_indexes, test_indexes