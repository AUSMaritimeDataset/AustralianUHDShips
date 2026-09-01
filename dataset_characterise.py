from pathlib import Path

from utils import labels_by_video, load_yolo_dets, yolo_dets_to_bbox_transform, BBOX_SIZE
from dji_video import VIDEO_FOLDER, KEYFRAME_FOLDER, FILE_TIMESTAMP_FORMAT
from typing import Any
from numpy.typing import NDArray
import numpy as np
from datetime import datetime
import cv2
import json
import yaml
import seaborn
import matplotlib.pyplot as plt

TARGET = 'all.txt'


def get_vid_len(vid: Path) -> float: # Seconds
    video = cv2.VideoCapture(vid)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()

    return frame_count / fps



TRACK_LEN = 4 + 2
def track_view(labels: list[Path], res: NDArray) -> dict[int, NDArray]:
    transform = yolo_dets_to_bbox_transform(res)
    tracks = {}
    ref_time = None
    for label in labels:
        time =datetime.strptime(label.stem, FILE_TIMESTAMP_FORMAT)
        if ref_time is None:
            ref_time = time
        dets = load_yolo_dets(label, transform=transform, force_dim=TRACK_LEN + 1, allow_missing=True)
        dets[:,-1] = (time - ref_time).total_seconds()
        for row in dets:
            obj_id = int(row[-2])
            if obj_id not in tracks:
                tracks[obj_id] = []
            tracks[obj_id].append(row)
    for track in tracks:
        tracks[track] = np.array(tracks[track])
    return tracks

NUM_BOXES = 'num_bboxes'
NUM_OBJ = 'num_obj'
AV_SIZE = 'av_size'
AV_SPEED = 'av_speed'
CLASS_COUNT = 'class_count'
CLASS = 'class'
VID_SEX = 'vid_secs'
MIN_SPEED = 'min_speed'
MAX_SPEED = 'max_speed'
MIN_SIZE = 'min_size'
MAX_SIZE = 'max_size'
NUM_FRAMES = 'num_frames'
def compute_object_stats(dets: NDArray) -> dict[str, float]:
    stats = {CLASS: dets[0][0]}
    stats[NUM_BOXES] = dets.shape[0]
    stats[AV_SIZE] = np.mean((dets[:,3] - dets[:,1]) * (dets[:,4] - dets[:,2]))
    speeds = []
    for i in range(1,len(dets)):
        row = dets[i]
        prev_row = dets[i-1]
        seconds = row[-1] - prev_row[-1]
        center = (row[1:3] + row[3:5]) / 2
        prev_center = (prev_row[1:3] + prev_row[3:5]) / 2
        speeds.append(np.linalg.norm(center - prev_center) / seconds)
    stats[AV_SPEED] = np.mean(np.array(speeds))
    return stats




SPEED_GRAPH = 'speeds.png'
SIZE_GRAPH = 'size.png'
CLASS_PIE = 'class_dist.png'
DPI = 50
def generate_figures(out: Path, speeds: list[float], sizes: list[float], class_counts: dict[int, int], class_map: dict[int, str]):


    speed_graph = seaborn.histplot(
        speeds,
        stat = 'proportion',
    )
    speed_graph.set_xlabel('Object Speed')
    speed_graph.set_ylabel('Count')
    speed_graph.set_title('Object Speed Distribution')
    speed_graph.get_figure().savefig(out / SPEED_GRAPH)

    size_graph = seaborn.histplot(
            sizes,
            stat = 'proportion',
    )
    size_graph.set_xlabel('Object Sizes')
    size_graph.set_ylabel('Count')
    size_graph.set_title('Object Sizes Distribution')
    size_graph.get_figure().savefig(out / SIZE_GRAPH)


    classes = []
    sums = []
    for cls, sum in class_counts.items():
        classes.append(class_map[cls])
        sums.append(sum)

    fig, ax = plt.subplots()
    ax.pie(
        sums, 
        labels=classes,
        autopct='%.0f%%'
    )
    fig.savefig(out / CLASS_PIE, bbox_inches='tight')






MANIFEST = 'manifest.yaml'
MAP = 'names'
STATIC_OUT = 'singular_numbers.json'
def characterise_dataset(root: Path, out: Path, res: NDArray):
    out.mkdir(exist_ok=True, parents=True)
    vid_folder = root / VIDEO_FOLDER
    keyframe_folder = root / KEYFRAME_FOLDER
    keyframe_file = keyframe_folder / TARGET
    manifest = keyframe_folder / MANIFEST
    with open(manifest, 'r', encoding='utf-8') as f:
        class_map = yaml.safe_load(f)[MAP]


    labels = labels_by_video(keyframe_file, keyframe_folder, tracks=True)

    num_frames = 0
    for vid in labels:
        num_frames += len(labels[vid])

    tracks = {vid: track_view(lab, res) for vid, lab in labels.items()}

   

    num_objects = 0
    num_boxes = 0
    class_counts = {}
    sizes = []
    speeds = []
    vid_len = (30*60) + 1 + (8*60) + 32 + (25*60) + 7 + (30*60) + (13*60) + 34 + (5*60) +  35 + (4*60)  + 19 + (28*60) + 10 + (34*60) + 26 
    for vid in tracks:
        vid_tracks = tracks[vid]
        # vid_len += get_vid_len((vid_folder / vid).with_stem('.MP4')) # Does fuck all

        stats = {track_id: compute_object_stats(dets) for track_id, dets in vid_tracks.items()}
        for track in stats:
            curr_stats = stats[track]
            num_objects += 1
            num_boxes += curr_stats[NUM_BOXES]
            cls = curr_stats[CLASS]
            if cls not in class_counts:
                class_counts[cls] = 0
            class_counts[cls] += 1
            size = curr_stats[AV_SIZE]
            if not np.isnan(size):
                sizes.append(size)
            speed = curr_stats[AV_SPEED]
            if not np.isnan(speed):
                speeds.append(speed)


    speeds= list(map(float, speeds))
    sizes = list(map(float, sizes))

    generate_figures(out, speeds, sizes, class_counts, class_map)

    with open(out / STATIC_OUT, 'w', encoding='utf-8') as f:
        json.dump({
            NUM_BOXES: num_boxes,
            NUM_OBJ: num_objects,
            VID_SEX: vid_len,
            MIN_SIZE: min(sizes),
            MAX_SIZE: max(sizes),
            MIN_SPEED: min(speeds),
            MAX_SPEED: max(speeds),
            NUM_FRAMES: num_frames
        }, f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Run characterisation on AustralianUHDShips')
    parser.add_argument('--root', '-i', type=str, required=True, help='Dataset root')
    parser.add_argument('--out', '-o', type=str, required=True, help='Output folder')
    args = parser.parse_args()

    HIGHRES = np.array([3840, 2160])
    characterise_dataset(Path(args.root).resolve(), Path(args.out).resolve(), HIGHRES)