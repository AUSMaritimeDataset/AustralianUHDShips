from pathlib import Path
from tempfile import TemporaryDirectory
import json
from datetime import datetime

from utils import get_params, TEST, load_model, YOLO, RTDETR
from dji_video import FILE_TIMESTAMP_FORMAT, VIDEO_EXT, dji_interval_generator
from zoom_sim import simulated_zoom_iterator, ZoomRegion, TimedImage
import re

TRACK = 'track'

INTERVAL_SIZE = 20
OVERLAP = True
INTERVAL_FOLDER = "interval"
INTERVAL_FOLDER_PADDING = 5

ZOOM_KEEP = 9
ZOOM_DATA = 'zoom_region.json'
LABEL_SUBFOLDER = 'labels'
LABEL_SUFFIX = '.txt'

ALPHA = 'alpha'
THETA = 'theta'

DATASET_NAME = 'name'
TARGET_RES = 'target_res'

def get_keytimes(label_file: Path) -> dict[str, list[datetime]]:
    with open(label_file, 'r', encoding='utf-8') as f:
        keyframes = f.readlines()
    key_times = {}
    for frame in keyframes:
        parts = frame.split('/')
        key_time = datetime.strptime(parts[-1].split('.')[0], FILE_TIMESTAMP_FORMAT)
        # Fall back video name from parent folder
        vid_name = label_file.parent.stem if len(parts) < 2 else parts[-2]
        if vid_name not in key_times:
            key_times[vid_name] = []
        key_times[vid_name].append(key_time)
        
    return key_times # Will be ordered as appears in file.


def track_interval( # Helper to ensure metadata is preserved
        model: YOLO | RTDETR, 
        tracker: Path, 
        interval: list[TimedImage], 
        out: Path, 
        device: int | str,
        index: int,
        zoom_region: ZoomRegion | None, 
        **kwargs
    ) -> Path:
    # Warmup fresh tracker
    with TemporaryDirectory() as temp:
        model.track(
            interval[0][1], 
            project=Path(temp).resolve(), # dumps garbage in last project
            device=device
        )

    # Track over interval
    frames = []
    rename_keytimes = []
    for time, frame in interval:
        frames.append(frame)
        rename_keytimes.append(time)
    results = model.track(
        frames, 
        device=device, 
        project = out, 
        name = f"{INTERVAL_FOLDER}_{str(index).rjust(INTERVAL_FOLDER_PADDING, "0")}", 
        persist=True,
        tracker = str(tracker),
        **kwargs
    )
    save_dir = Path(results[0].save_dir).resolve()
    del results # Only wanted directory

    # Log metadata required for val
    images = sorted(list(save_dir.glob('image*.*')), key=lambda s: [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', s.stem)])
    labels = [(save_dir / LABEL_SUBFOLDER / im.name).with_suffix(LABEL_SUFFIX) for im in images]
    assert len(images) == len(rename_keytimes)
    for image, label, keytime in zip(images, labels, rename_keytimes):
        image.rename(image.with_stem(datetime.strftime(keytime, FILE_TIMESTAMP_FORMAT)))
        if label.is_file():
            label.rename(label.with_stem(datetime.strftime(keytime, FILE_TIMESTAMP_FORMAT)))
    if zoom_region is not None:
        alpha, theta = zoom_region
        with open(save_dir / ZOOM_DATA, 'w', encoding='utf-8') as f:
            json.dump({
                ALPHA: alpha,
                THETA: theta,
            }, f, indent = 4)
    return save_dir

def eval_tracker(
    checkpoints: list[Path], 
    label_file: Path,
    video_folder: Path, 
    trackers: list[Path],
    out_dir: Path, 
    config: Path | None, 
    device: int | str, 
    zoom: bool = False,
    zoom_tours: Path | None = None,
    **kwargs
):

    # prepare benchmark parameters
    track_params = get_params(config, mode=TEST, **kwargs)
    track_params.update(get_params(config, mode=TRACK))
    track_params.update({ # Enforce benchmark behavior
        "save": True, 
        "save_txt": True, 
        "save_conf": True,
    })
    dataset_name = track_params.pop(DATASET_NAME, "")
    target_res = track_params.pop(TARGET_RES, None)
    assert target_res is not None or not zoom
    keytimes = get_keytimes(label_file)

    # Load tours if relevant
    if zoom_tours is not None:
        with open(zoom_tours, 'r', encoding='utf-8') as f:
            zoom_tours = json.load(f) # TODO Write tours somewhere
            
    for checkpoint in checkpoints:
        # Set up output
        run_name = checkpoint.stem
        if dataset_name:
            run_name += f"_{dataset_name}"
        run_parent_dir = out_dir / run_name

        model = load_model(checkpoint)

        for tracker in trackers:
            run_dir = run_parent_dir / tracker.stem
            for vid in keytimes:
                # Set up video output
                vid_out = run_dir / vid
                vid_out.mkdir(parents=True, exist_ok=False)

                if zoom:
                    tour = zoom_tours[vid] if zoom_tours is not None else zoom_tours
                    for i, zoom_interval in enumerate(simulated_zoom_iterator(video_folder / (vid + VIDEO_EXT),
                                            keytimes[vid], INTERVAL_SIZE, ZOOM_KEEP, target_res,
                                            overlap=OVERLAP, zoom_regions=tour, label_file=image_file, return_region=True)):
                        interval, region = zoom_interval
                        track_interval(
                            model,
                            tracker,
                            interval,
                            vid_out,
                            device,
                            i,
                            region,
                            **track_params
                        )     
                        
                else:
                    for i, interval in enumerate(dji_interval_generator(video_folder / (vid + VIDEO_EXT), 
                                            keytimes[vid], INTERVAL_SIZE, OVERLAP)):
                        track_interval(
                            model,
                            tracker,
                            interval,
                            vid_out,
                            device,
                            i,
                            None,
                            **track_params
                        )
                        
                        

if __name__ == "__main__":
    # CLI to train detectors
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Evaluate models on dataset.')
    parser.add_argument('--weights', '-w', type=str, nargs='+', required=True, help='Models to benchmark')
    parser.add_argument('--image_file', '-i', type=str, required=True, help='Path to image file.')
    parser.add_argument('--trackers', '-t', type=str, nargs='+', required=True, help='Trackers to benchmark')
    parser.add_argument('--videos', '-v', type=str, required=True, help='Folder containing videos to build intervals from')
    parser.add_argument('--config', '-c', type=str, default='', help='Path to config.')
    parser.add_argument('--out', '-o', type=str, required=True, help='Path to output folder.')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device to run inference on (e.g., 0, 1, cpu).')
    parser.add_argument('--zoom_tours', '-z', type=str, default='', help='Standard tours for zoom')
    parser.add_argument('--no_zoom', '-n', action='store_true', help='No simulated zoom operation')
    parser.set_defaults(no_zoom=False)
    args = parser.parse_args()

    # parse the parsel. I am so funny and hot
    device = args.device
    if device != 'cpu':
        device = int(device)
    checkpoints = [Path(checkpoint).resolve() for checkpoint in args.weights]
    image_file = Path(args.image_file).resolve()
    vids = Path(args.videos).resolve()
    config = Path(args.config).resolve() if args.config else None
    out = Path(args.out).resolve()
    zoom_tours = Path(args.zoom_tours) if args.zoom_tours else None
    evaluate_zoom = not args.no_zoom
    trackers = [Path(tracker_config).resolve() for tracker_config in args.trackers]

    eval_tracker(checkpoints, image_file, vids, trackers, out, config, 
                 device, evaluate_zoom, zoom_tours)