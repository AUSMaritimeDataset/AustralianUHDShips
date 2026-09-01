from pathlib import Path
from tempfile import TemporaryDirectory
from scipy.io import loadmat
import cv2
import numpy as np

from conversion_utils import write_yolo, train_val_test_split, TRAIN, VAL, TEST

SINGAPORE_STRUCT = 'structXML'
SING_CLASS = 'ObjectType'
SING_BBOX = 'BB'
VID_FOLDER = 'Videos'
LABEL_FOLDER = 'ObjectGT'
LABEL_SUFFIX = '.mat'
VID_SUFFIX = '.avi'
VID_END = 'OB'
FRAME_FORMAT = '.png'

PARTS = ('VIS_Onboard', 'VIS_Onshore')
WIDTH, HEIGHT = 1920, 1080 # Fixed image size

SPLIT_SEED = 69420
VAL_PROP = 0.15
TEST_PROP = 0.15

def singapore_to_yolo(singapore_root: Path, out_dir: Path):

    reverse_map = {} # Need to track present classes as we see them
    next_class = 0

    images = []
    labels = []
    with TemporaryDirectory() as t: # Need to extract frames from videos
        working = Path(t).resolve()
        for dataset_part in PARTS:
            # Useful locations
            part_folder = singapore_root / dataset_part
            vid_folder = part_folder / VID_FOLDER
            label_folder = part_folder / LABEL_FOLDER
            for label_file in label_folder.glob('*'+LABEL_SUFFIX):
                # Load labels
                with open(label_file, 'rb') as f:
                    label_data = loadmat(f, simplify_cells=True)[SINGAPORE_STRUCT]

                # Load video
                vid_name = (label_file.stem.removesuffix(f"_{LABEL_FOLDER}")) + VID_SUFFIX
                vid = vid_folder / vid_name
                capture = cv2.VideoCapture(str(vid))

                # Extract frames and labels
                print(f"Covnerting {vid.stem}...")
                extraction_space = working / vid.stem
                extraction_space.mkdir(parents=True)
                frame_index = -1
                while capture.isOpened():
                    frame_exists, curr_frame = capture.read()
                    if not frame_exists:
                        break
                    else:
                        frame_index += 1
                    if frame_index >= len(label_data):
                        break

                    # Copy image
                    image_path = extraction_space / f"frame_{frame_index}{FRAME_FORMAT}"
                    cv2.imwrite(str(image_path), curr_frame)

                    # Load labels
                    frame_label = label_data[frame_index]
                    classes = np.array(frame_label[SING_CLASS], ndmin=1)
                    bboxes = np.array(frame_label[SING_BBOX], ndmin=2) # Load ensuring homogenous structure
                    label_entry = []
                    for cls, bbox in zip(classes, bboxes):
                        if not type(cls) == str:
                            label_entry = None
                            break # Corrupt label
                        if cls not in reverse_map: # Record classes as we see them
                            reverse_map[cls] = next_class
                            next_class += 1
                        cls_entry = reverse_map[cls]
                        l,t,w,h = bbox # top left corner fixation
                        x = (l + (w / 2)) / WIDTH # Normalise
                        y = (t + (h / 2)) / HEIGHT
                        w /= WIDTH
                        h /= HEIGHT

                        label_entry.append((cls_entry,x,y,w,h))

                    # Register entry
                    if label_entry is not None:
                        images.append(image_path)
                        labels.append(label_entry)

                capture.release()

        assert len(images) == len(labels)
        # Split images up into split deterministically
        train, val, test = train_val_test_split(
            len(images), 
            val_prop= VAL_PROP,
            test_prop= TEST_PROP,
            seed=SPLIT_SEED
        )
        split_images = {TRAIN: [], VAL: [], TEST: []}
        split_labels = {TRAIN: [], VAL: [], TEST: []}
        for i, entry in enumerate(zip(images, labels)):
            im, lab = entry
            if i in train:
                split_images[TRAIN].append(im)
                split_labels[TRAIN].append(lab)
            elif i in val:
                split_images[VAL].append(im)
                split_labels[VAL].append(lab)
            elif i in test:
                split_images[TEST].append(im)
                split_labels[TEST].append(lab)
            else:
                raise ValueError(f"{i} does not appear in split")

        # reverse class map (keys will be unique)
        class_map = {val: key for key, val in reverse_map.items()}

        write_yolo(split_images, split_labels, out_dir, class_map)


if __name__ == "__main__":
    # CLI
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Convert SingaporeMarine to YOLO format.')
    parser.add_argument('--input', '-i', type=str, required=True, help='SingaporeMarine root dir')
    parser.add_argument('--out', '-o', type=str, required=True, help='Output converted dir.')
    args = parser.parse_args()

    # Parse paths
    root = Path(args.input).resolve()
    out = Path(args.out).resolve()

    singapore_to_yolo(root, out)