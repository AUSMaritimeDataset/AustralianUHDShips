from pathlib import Path
import json
from tqdm import tqdm

from conversion_utils import write_yolo, SPLITS


SEACLIPS = "seaclips"
IMAGES = "images"
FILE_NAME = "file_name"
IMG_ID = "id"
ANNOT_ID = "image_id"
ANNOTATIONS = "annotations"
CLASS = "category_id"
WIDTH, HEIGHT = "width", "height"
BBOX = "bbox"
CLASSES = "categories"
CLASS_ID = "id"
CLASS_NAME = "name"
def seaclips_to_yolo(seaclips_root: Path, out_dir: Path):
    images = {}
    labels = {}
    for split in SPLITS:
        # load seaclips format
        seaclips_manifest = seaclips_root / f"{SEACLIPS}-{split}.json"
        with open(seaclips_manifest, 'r') as f:
            split_data = json.load(f)

        # Determine images
        images[split] = []
        frame_map = {}
        sizes = []
        print(f"Loading {split} images...")
        for i, img in tqdm(enumerate(split_data[IMAGES])):
            images[split].append(seaclips_root / split / img[FILE_NAME])
            assert img[IMG_ID] not in frame_map 
            frame_map[img[IMG_ID]] = i # Allow mapping labels to correct index
            sizes.append((img[WIDTH], img[HEIGHT]))

        # Extract labels
        print(f"Loading {split} labels...")
        labels[split] = [[] for _ in range(len(images[split]))] # Want to drop in as we see things
        for label in tqdm(split_data[ANNOTATIONS]):
            destination_index = frame_map[label[ANNOT_ID]]
            x,y,w,h = label[BBOX]
            x = x + (w / 2) # COCO uses TL corner not center
            y = y + (h / 2)
            x /= sizes[destination_index][0] # Normalise
            y /= sizes[destination_index][1]
            w /= sizes[destination_index][0]
            h /= sizes[destination_index][1]

            new_entry = (
                round(label[CLASS])-1,
                x,y,w,h
            )

            labels[split][destination_index].append(new_entry)

    class_map = {cls[CLASS_ID]-1: cls[CLASS_NAME] for cls in split_data[CLASSES]}
    write_yolo(images, labels, out_dir, class_map)

    

if __name__ == "__main__":
    # CLI
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Convert Seaclips to YOLO format.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Seaclips root dir')
    parser.add_argument('--out', '-o', type=str, required=True, help='Output converted dir.')
    args = parser.parse_args()

    # Parse paths
    root = Path(args.input).resolve()
    out = Path(args.out).resolve()

    seaclips_to_yolo(root, out)
