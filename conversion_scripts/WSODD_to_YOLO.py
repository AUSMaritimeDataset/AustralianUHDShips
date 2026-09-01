from pathlib import Path
from tqdm import tqdm
from xml.etree import ElementTree

from conversion_utils import write_yolo, TRAIN, VAL, TEST, train_val_test_split

IMAGE_FOLDER = "image"
LABEL_FOLDER = "annotation"
LABEL_SUFFIX = ".xml"

SPLITSEED = 42069
VAL_PROP = 0.15
TEST_PROP = 0.15

SIZE_TAG = 'size'
WIDTH_TAG = 'width'
HEIGHT_TAG = 'height'
ANNOTATIONS_TAG = 'object'
CLASS_TAG = 'name'
BBOX_TAG = 'bndbox'
XMIN_TAG = 'xmin'
XMAX_TAG = 'xmax'
YMIN_TAG = 'ymin'
YMAX_TAG = 'ymax'

def wsodd_to_yolo(wsodd_root: Path, out_dir: Path):
    # Identify available images
    images = list((wsodd_root / IMAGE_FOLDER).glob('*'))

    # Split images up into split deterministically
    train, val, test = train_val_test_split(
        len(images), 
        val_prop= VAL_PROP,
        test_prop= TEST_PROP,
        seed=SPLITSEED
    )
    split_images = {TRAIN: [], VAL: [], TEST: []}
    for i, path in enumerate(images):
        if i in train:
            split_images[TRAIN].append(path)
        elif i in val:
            split_images[VAL].append(path)
        elif i in test:
            split_images[TEST].append(path)
        else:
            raise ValueError(f"{i} does not appear in split")

    # Convert labels for each image
    labels = {}
    reverse_map = {} # Need to construct class map as we go
    next_label = 0
    for split in split_images:
        # Determine label files
        label_files = []
        for image_path in split_images[split]:
             # File names include . because I am not allowed to be happy
            file_name = str(image_path).rsplit('/',1)[-1].rsplit('.',1)[0]
            label_name = file_name + LABEL_SUFFIX
            label_files.append(image_path.parent.parent / LABEL_FOLDER / label_name)

        # Parse label files for each image
        labels[split] = []
        print(f"Loading {split} labels...")
        for label_file in tqdm(label_files):
            label_root = ElementTree.parse(str(label_file)).getroot()

            # Image data for normalisation
            size_data = label_root.find(SIZE_TAG)
            width = int(size_data.find(WIDTH_TAG).text)
            height = int(size_data.find(HEIGHT_TAG).text)

            # Object Annotations
            annotations = label_root.findall(ANNOTATIONS_TAG)
            converted_labels = []
            for annotation in annotations:
                cls = annotation.find(CLASS_TAG).text
                if cls not in reverse_map: # Stash existing classes on the way past
                    reverse_map[cls] = next_label
                    next_label += 1
                cls = reverse_map[cls]

                # Determine normalised bbox
                bbox = annotation.find(BBOX_TAG)
                xmin = int(bbox.find(XMIN_TAG).text)
                xmax = int(bbox.find(XMAX_TAG).text)
                ymin = int(bbox.find(YMIN_TAG).text)
                ymax = int(bbox.find(YMAX_TAG).text)
                x = ((xmin + xmax) / 2) / width
                y = ((ymin + ymax) / 2) / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                converted_labels.append((cls,x,y,w,h))

            labels[split].append(converted_labels)

    # reverse class map (keys will be unique)
    class_map = {val: key for key, val in reverse_map.items()}
    write_yolo(split_images, labels, out_dir, class_map)

    

if __name__ == "__main__":
    # CLI
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Convert WSODD to YOLO format.')
    parser.add_argument('--input', '-i', type=str, required=True, help='WSODD root dir')
    parser.add_argument('--out', '-o', type=str, required=True, help='Output converted dir.')
    args = parser.parse_args()

    # Parse paths
    root = Path(args.input).resolve()
    out = Path(args.out).resolve()

    wsodd_to_yolo(root, out)
