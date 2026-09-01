from pathlib import Path
from typing import Iterator
from numpy.typing import NDArray
from datetime import datetime, timedelta
import cv2

FRAME_EXT = ".png"
VIDEO_EXT = ".MP4"

LOG_INTERVAL = 1000
VIDEO_FOLDER = 'videos'
KEYFRAME_FOLDER = 'keyframes'
IMAGE_FOLDER = 'images'
KEYFRAME_MANIFEST = "benchmark.txt"

DJI_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
FILE_TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S_%f" # For labelled filenames

TimedImage = tuple[datetime, NDArray]

extract_dji_datetime = lambda vid: datetime.strptime(vid.stem.split("_")[1], DJI_TIMESTAMP_FORMAT)

def timed_frames_from_dji_video(vid: Path) -> Iterator[TimedImage]:
    start_time = extract_dji_datetime(vid)
    cap = cv2.VideoCapture(str(vid))

    frame_no = 0
    print("Processing DJI Video Frames. This could take a hot minute...")
    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if not frame_exists:
            break

        time_from_start = timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC))
        timestamp = start_time + time_from_start
        
        yield (timestamp, curr_frame)

        frame_no += 1
        if not frame_no % LOG_INTERVAL: 
            print(f"Processed {frame_no} frames") 

    cap.release()
    print("Done!")

def sparse_dji_iterator(
        vid: Path, 
        keytimes: list[datetime], 
        keytime_radius: float | tuple[float,float]
) -> Iterator[TimedImage]:
    start_time = extract_dji_datetime(vid)
    # Set up desired radius for arithmetic
    if isinstance(keytime_radius, float):
        keytime_radius = (keytime_radius, keytime_radius)
    keytime_radius = (timedelta(seconds=keytime_radius[0]), timedelta(seconds=keytime_radius[1]))
    cap = cv2.VideoCapture(str(vid))

    keytimes = sorted(keytimes) # Copy to not consume list for other purposes
    to_process = len(keytimes)
    if not to_process:
        print("Warning: Empty Keytime list")
        return
    curr_keyframe = keytimes.pop(0)
    lb = curr_keyframe - keytime_radius[0]
    ub = curr_keyframe + keytime_radius[1]
    processed = 0
    print("Processing DJI Video Frames...")
    
    prev = None
    while(cap.isOpened()):
        cap.grab() # Advance without necessarily decoding frame
        time_from_start = timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC))
        timestamp = start_time + time_from_start
        if timestamp == prev:
            break # grab seems to get stuck on final frame in some odd edge case
        prev = timestamp

        if timestamp < lb:
            continue
        elif timestamp > ub:
            processed += 1
            print(f"Processed {processed}/{to_process} keyframes")
            if not keytimes:
                break
            curr_keyframe = keytimes.pop(0)    
            lb = curr_keyframe - keytime_radius[0]
            ub = curr_keyframe + keytime_radius[1]    
        if lb <= timestamp <= ub: # Could overlap
            frame_exists, curr_frame = cap.retrieve()
            if not frame_exists:
                break
            yield (timestamp, curr_frame)

    cap.release()

def dji_interval_generator(
        vid: Path, 
        keytimes: list[datetime], 
        interval_size: int, 
        overlap: bool = True
) -> Iterator[tuple[TimedImage]]:
    SPARSE_RADIUS = 0.1 # To allow interval times that don't fall exactly on the frame

    keytimes = sorted(keytimes) # Copy to not consume list for other purposes
    if len(keytimes) < 2 or interval_size < 2:
        print("Warning: Not enough keytimes for interval")
        return

    # Compute infilled times in advance for spare iterator
    middle_count = interval_size - 2
    compute_delta = lambda start, stop: (stop - start) / (middle_count + 1)
    infill_keypoints = keytimes.copy()
    start = infill_keypoints.pop(0)
    infilled = [start]
    while infill_keypoints:
        stop = infill_keypoints.pop(0) # Ensure we capture final interval
        delta = compute_delta(start, stop)
        middle_points = [start + (delta*(i+1)) for i in range(middle_count)]
        infilled += middle_points
        infilled.append(stop)

        # move to new interval
        if overlap:
            start = stop
        else:
            start = infill_keypoints.pop(0)
            infilled.append(start)

    start = keytimes.pop(0)
    stop = keytimes.pop(0)
    delta = compute_delta(start, stop)
    interval = []
    target = start
    for timed_frame in sparse_dji_iterator(vid, infilled, SPARSE_RADIUS):
        time = timed_frame[0]
        if time < target:
            continue # Scan to next keyframe
        else:
            if time > target and target in (start, stop):
                raise ValueError(f"Provided Keytime {target} does not exist in {vid}")
            interval.append(timed_frame)

            if len(interval) == interval_size:
                yield interval
                if not keytimes:
                    break
                # New interval
                interval = []
                new_start = keytimes.pop(0)
                if overlap:
                    start = stop
                    stop = new_start
                    interval.append(timed_frame)
                else:
                    if not keytimes:
                        break
                    start = new_start
                    stop = keytimes.pop(0)
                delta = compute_delta(start, stop)

            curr_size = len(interval)
            target = stop if curr_size == (interval_size - 1) else start + (delta * curr_size)


# TODO docstrings
def extract_keyframes(vid: Path, out: Path, interval_seconds: float = 10):
    # Prep and prevent overwrite
    manifest = out / KEYFRAME_MANIFEST
    out.mkdir(parents=True, exist_ok=False) 
    manifest.touch(exist_ok=False)

    interval = timedelta(seconds=interval_seconds)

    gap_start = extract_dji_datetime(vid)
    for time, frame in timed_frames_from_dji_video(vid):
        if time >= gap_start + interval:
            # write keyframe
            filename = time.strftime(FILE_TIMESTAMP_FORMAT) + FRAME_EXT
            cv2.imwrite(out / filename, frame)
            with open(manifest, 'a', encoding='utf-8') as f:
                f.write(filename + '\n') 
            
            gap_start = time

def collate_keyframe_manifest(folder: Path):
    image_folder = folder / IMAGE_FOLDER
    keyframes = []
    for keyframe_folder in image_folder.glob('*'):
        manifest = keyframe_folder / KEYFRAME_MANIFEST
        with open(manifest, 'r', encoding='utf-8') as f:
            curr_keyframes = f.readlines()
        keyframes += [f'{IMAGE_FOLDER}/{keyframe_folder.stem}/{keyframe}' for keyframe in curr_keyframes]
    with open(folder/KEYFRAME_MANIFEST, 'w+', encoding='utf-8') as f:
        f.writelines(keyframes)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from tqdm import tqdm
    parser = ArgumentParser(description='Extract Keyframes from DJI Videos')
    parser.add_argument('--input', '-i', type=str, required=True, help='Folder containing videos to process')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output folder')
    parser.add_argument('--keyframe_interval', '-t', type=float, default=10, help='Seconds between designated keyframes')
    args = parser.parse_args()
    
    vid_folder = Path(args.input).resolve()
    out_folder = Path(args.output).resolve()

    for vid in tqdm(vid_folder.glob('*.MP4')):
        out = out_folder / vid.stem

        extract_keyframes(vid, out, args.keyframe_interval)
    collate_keyframe_manifest(out_folder)