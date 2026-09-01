from pathlib import Path
import json
from ultralytics import RTDETR, YOLO
from numpy.typing import NDArray
import numpy as np

TRAIN, TEST = "train", "test"
MODES = (TRAIN, TEST)
BBOX_SIZE = 4

def load_model(checkpoint: Path) -> RTDETR | YOLO:
    """
    Load correct ultralytics model based on checkpoint name.

    Args:
        checkpoint (Path): path to checkpoint

    Preconditions:
        checkpoint name contains `rt` (Case insensitive) if and only if model is 
                an RT-DETR model.
    Returns:
        RTDETR | YOLO: Loaded model
    """
    RTDETR_FINGERPRINT = "rt"
    name = checkpoint.stem.lower()
    if RTDETR_FINGERPRINT in name:
        return RTDETR(str(checkpoint))
    else:
        return YOLO(str(checkpoint))

def load_config(config: Path, mode: str | None = None) -> dict:
    """
    Helper to abstract away configs that may cover multiple modes.

    Args:
        config (Path): Config to load
        mode (str | None): Named mode to load, or None to load flat.

    Returns:
        dict: Loaded config parameters
    """
    with open(config, 'r', encoding='utf-8') as f:
        params = json.load(f)
    if mode is not None and mode in params:
        return params[mode]
    elif mode is not None:
        print("Warning: Mode not specified within config, loading Flat")
    return params

def get_params(config: Path | None, mode: str = TRAIN, **overrides) -> dict:
    """
    Helper that enables loading params from config and/or arguments. 

    Args:
        config (Path | None): config to load from
        **overrides can also be specified and these take precedent.
    """
    params = {}
    if config is not None:
        params.update(load_config(config, mode))
    params.update(overrides) # Support Overrides
    return params

def yolo_dets_to_bbox_transform(res: NDArray) -> NDArray: # [x,y,w,h] -> [x1,y1,x2,y2]
    scalor = np.diag(res)
    convertor = np.r_[
        np.c_[scalor, scalor],
        np.c_[-scalor, scalor] // 2
    ]
    return convertor

def images_by_video(label_file: Path, root: Path | None = None) -> dict[str, list[Path]]:
    if root is None:
        root = label_file.parent # Common assumption

    with open(label_file, 'r', encoding='utf-8') as f:
        keyframes = f.readlines()
    
    images = {}
    for frame in keyframes:
        parts = frame.split('/')
        # Fall back video name from parent folder
        vid_name = root.stem if len(parts) < 2 else parts[-2]
        if vid_name not in images:
            images[vid_name] = []
        images[vid_name].append(root / frame)
    return images

def labels_by_video(
        labels: Path | dict[str, list[Path]], 
        root: Path | None = None, tracks: bool = False
) -> dict[str, list[Path]]:
    
    LABEL_SUFFIX = '.txt'
    IMAGE_FOLDER = 'images' # Assume ultralytics structure
    LABEL_FOLDER = 'labels'
    TRACK_FOLDER = 'tracks' # Our structure

    if isinstance(labels, Path):
        labels = images_by_video(labels, root)
    
    to_load = TRACK_FOLDER if tracks else LABEL_FOLDER
    return {vid: [
        Path(str(im).replace(IMAGE_FOLDER, to_load)).with_suffix(LABEL_SUFFIX).resolve() 
        for im in images] for vid, images in labels.items()}

def load_yolo_dets(
        csv: Path, 
        transform: NDArray | None = None,
        force_dim: int = 0,
        allow_missing: bool = False,
) -> NDArray:
    """
    Load detections from ultralytics det file.  

    Args:
        csv (Path): CSV to load from
        transform (NDArray): transform to apply to bbox before returning.
        force_dim (force_dim): force empty arrays to have specified width or 0 to disable.
        allow_missing (bool): Have missing files be loaded as empty.

    Returns:
        NDArray: Loaded detections, cls, x,y,w,h (conf, id) if no transform
    """
    if transform is None:
        transform = np.identity(BBOX_SIZE)
    try:
        dets = np.genfromtxt(csv, delimiter=' ',ndmin=2)
    except FileNotFoundError as e:
        if not allow_missing:
            raise e
        dets = np.ndarray((0, 0))
    if dets.shape[0] == 0 and force_dim:
        dets = np.ndarray((0, force_dim))
    if dets.shape[1] < force_dim:
        if allow_missing:
            padding = np.ones((dets.shape[0], force_dim)) * -1 # -1 indicates junk data here
            padding[:, :dets.shape[1]] = dets
            dets = padding
        else:
            raise ValueError("Insufficient information to fill dims")

    dets[:, 1:BBOX_SIZE+1] = np.matmul(dets[:, 1:BBOX_SIZE+1], transform)
    return dets

