from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import yaml
import json
from tqdm import tqdm
from copy import deepcopy

from utils import TEST, load_config, labels_by_video, yolo_dets_to_bbox_transform, load_yolo_dets, BBOX_SIZE
from eval_tracker import LABEL_SUBFOLDER, ZOOM_DATA, ALPHA, THETA, LABEL_SUFFIX, TRACK

EVAL = 'eval'
TRANS = "translation"
NAME = "name"
RES = "res"
LABEL_ROOT = 'path'
CLASS_MAP = 'names'
TARGET_RES = 'target_res'

JSON = '.json'
MEAN = 'mean'

THRESHOLDS = (0.25, 0.5, 0.7, 0.95)
AV_OVER_THRESHOLDS = np.linspace(0.5,0.95,10)

POS, CLS = 'pos', 'class'
MEAN = 'mean'
CROPPED_PREFIX = 'cropped_'
PRECISION =  'precision'
RECALL = 'recall'
F1 = 'f1'
AP = 'ap'
AV_CONF_CORR = 'av_corr_conf'
AV_CONF_ERR = 'av_err_conf'
LABEL_SEP = '@'

MISSED = 'never_detected'
DROPPED = 'dropped'
MAINTAINED = 'tracked'
PROPORTIONAL_SUFFIX = '_of_detected'
BLANK = 'blank'
WITH = 'with'
WITHOUT = 'without'

# The function count kind of popped off so I leant into it
intersection = lambda bbox1, bbox2: (max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), 
                                     min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3]))
def iou(bbox1:NDArray, bbox2: NDArray):
    """
    Compute iou between two bboxes
    """
    area = lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    bbox_int = intersection(bbox1, bbox2)
    if bbox_int[2] < bbox_int[0]  or bbox_int[3] < bbox_int[1]:
        return 0
    else:
        return area(bbox_int) / (area(bbox1) + area(bbox2) - area(bbox_int))

def crop_detections(detections: NDArray, region: NDArray) -> NDArray:
    """
    crop detections to region
    """
    culled_dets = detections[[iou(detections[i, 1:BBOX_SIZE+1], region) > 0 for i in range(detections.shape[0])]]
    if culled_dets.shape[0] > 0:
        culled_dets[:, 1:BBOX_SIZE+1] = np.array([intersection(culled_dets[i, 1:BBOX_SIZE+1], region) for i in range(culled_dets.shape[0])])
    return culled_dets

def match_objects(
        predictions: NDArray, 
        ground_truth: NDArray, 
        threshold: float = 0.5,
        cls: int | None = None, # None means all
        ignore_det_class: bool = False,
) -> tuple[dict[int, int], tuple[float,float,float]]: # Assoc, TP,FP,FN
    """
    Associate detections with ground truth,

    Args:
        predictions (NDArray): Predictions to associate.
        ground_truth (NDArray): Ground truth detections.
        threshold (float, optional): IOU threshold. Defaults to 0.5.
        cls (int | None, optional): Class to restrict to. Defaults to None.

    Returns:
        tuple[dict[int, int], tuple[float,float,float]]: Associations (GT: pred) 
                and corresponding summative metrics (True pos, False pos, False neg)
    """
    if cls is not None:
        ground_truth = ground_truth[ground_truth[:,0] == cls]
        if not ignore_det_class:
            predictions = predictions[predictions[:,0] == cls]

    fp = predictions.shape[0]
    fn = ground_truth.shape[0]
    tp = 0
    matched = set()
    assoc = {}
    for i, object in enumerate(ground_truth):
        det = None
        best_overlap = 0
        for j, pred in enumerate(predictions):
            if j not in matched:
                pred_iou = iou(object[1:BBOX_SIZE+1], pred[1:BBOX_SIZE+1])
                if pred_iou >= max(best_overlap, threshold):
                    det = j
                    best_overlap = pred_iou

        if det is not None:
            matched.add(det)    
            tp += 1
            fp -=1
            fn -=1
        assoc[i] = det

    return (assoc, (tp,fp,fn))

precision_recall = lambda tp, fp, fn: ( # (precision, recall)
    tp / (tp + fp) if tp+fp else 1, tp / (tp + fn) if tp+fn else 1
)
f1_score = lambda prec, rec: (2*prec*rec) / (prec+ rec) if prec+rec else 0

def average_precision(
    predictions: NDArray, 
    ground_truth: NDArray,
    threshold: float = 0.5,
    cls: int | None = None, # None means all
    ignore_det_class: bool = False,
):
    """
    Compute AP score using decreasing recall method with standard interpolated 
    recall.

    Args:
        predictions (NDArray): Predictions to evaluate.
        ground_truth (NDArray): Ground truth detections
        threshold (float, optional): IOU Threshold. Defaults to 0.5.
        cls (int | None, optional): Class to restrict to. Defaults to None.
    """

    subset_score = lambda sub: precision_recall(*(match_objects(sub, ground_truth, threshold, cls, ignore_det_class)[1]))

    if cls is not None and not ignore_det_class:
        predictions = predictions[predictions[:,0] == cls]

    confidence_order = predictions[np.argsort(predictions[:, BBOX_SIZE+1])[::-1]]
    prev_prec, prev_rec = subset_score(confidence_order)
    score = 0
    for i in range(1,confidence_order.shape[0]): # Need to recompute each time due to potential overlaps, guh
        precision, recall = subset_score(confidence_order[i:])
        if precision > prev_prec:
            score += prev_prec * (prev_rec - recall)
            prev_prec, prev_rec = precision, recall
    score += prev_prec * prev_rec # Extend head of curve to ensure area
    return score

def aggregate_metrics(
        predictions: NDArray, 
        ground_truth: NDArray,
        threshold: float,
        cls: int | None = None
) -> dict[str, float]:
    """
    Wrapper to compute all aggregate metrics for a given threshold and class.
    """
    t = int(threshold*100)
    associated, pairing_metrics = match_objects(predictions, ground_truth, threshold, cls)
    precision, recall = precision_recall(*pairing_metrics)
    f1 = f1_score(precision, recall)
    ap = average_precision(predictions, ground_truth, threshold, cls)
    assoc_ids = set(j for j in associated.values() if j is not None)
    correct_confs = []
    incorrect_confs = []
    for j,det in enumerate(predictions):
        if j in assoc_ids:
            correct_confs.append(det[BBOX_SIZE+1])
        else: 
            incorrect_confs.append(det[BBOX_SIZE+1])
    return {
        f'{PRECISION}{LABEL_SEP}{t}': precision,
        f'{RECALL}{LABEL_SEP}{t}': recall,
        f'{F1}{LABEL_SEP}{t}': f1,
        f'{AP}{LABEL_SEP}{t}': ap,
        f'{AV_CONF_CORR}{LABEL_SEP}{t}': (sum(correct_confs)/len(correct_confs)) if len(correct_confs) else 0,
        f'{AV_CONF_ERR}{LABEL_SEP}{t}': (sum(incorrect_confs)/len(incorrect_confs)) if len(incorrect_confs) else 0
    }

def get_by_prefix(d: dict, prefix: str): # very inefficient but I don't care
    """
    Helper to key dictionary by prefix
    """
    for key in d.keys():
        if key.startswith(prefix):
            return d[key]

def compute_base_metrics(
        predictions: NDArray, 
        ground_truth: NDArray,
        cls: int | None = None,
) -> dict[str,float]:
    """
    Wrapper to compute metrics across threshold range
    """
    metrics = {}
    for thresh in THRESHOLDS:
        metrics |= aggregate_metrics(predictions,ground_truth,thresh,cls)
    
    av_components = [aggregate_metrics(predictions, ground_truth,t,cls) for t in AV_OVER_THRESHOLDS]
    tags = [key.split(LABEL_SEP, 1)[0] for key in av_components[0].keys()]  
    new_suffix = f'{LABEL_SEP}{int(AV_OVER_THRESHOLDS[0]*100)}:{int(AV_OVER_THRESHOLDS[-1]*100)}'
    av_metrics = {(tag + new_suffix): [] for tag in tags}
    # Hyper inefficient but I have no more time to care
    for component in av_components:
        for tag in tags:
            get_by_prefix(av_metrics, tag).append(get_by_prefix(component, tag))
    for key in av_metrics:
        av_metrics[key] = sum(av_metrics[key]) / len(av_metrics[key])
    return metrics | av_metrics

def dict_average(
    dicts: list[dict] # Should have same keys
):
    """
    Helper to compute average of all dict keys
    """
    average = {}
    keys = dicts[0].keys()
    for key in keys:
        vals = [d.get(key, 0) for d in dicts]
        if isinstance(vals[0], dict):
            average[key] = dict_average(vals)
        else:
            average[key] = sum(vals)/len(vals) 
    return average

def compute_full_metrics(
        predictions: NDArray, 
        ground_truth: NDArray,
        roi: NDArray | None,
        class_map: dict[int, str],
) -> dict[str, float]:
    """
    Wrapper to compute metrics across classes
    """
    crop = roi is not None

    metrics = {POS: {}, CLS: {}}
    
    # Pure pos
    uncropped = compute_base_metrics(predictions, ground_truth)
    metrics[POS] |= uncropped 

    if crop:
        cropped_pred, cropped_truth = crop_detections(predictions, roi), crop_detections(ground_truth, roi)
        cropped = compute_base_metrics(cropped_pred, cropped_truth)
        metrics[POS] |= {
            CROPPED_PREFIX + key: val for key,val in cropped.items()
        }

    # Class Aware
    for cls, name in class_map.items():
        uncropped = compute_base_metrics(predictions, ground_truth, cls)
        metrics[CLS][name] = uncropped 

        if crop:
            cropped = compute_base_metrics(cropped_pred, cropped_truth, cls)
            metrics[CLS][name] |= {
                CROPPED_PREFIX + key: val for key,val in cropped.items()
            }

    metrics[CLS][MEAN] = dict_average(
        [metrics[CLS][cls] for cls in class_map.values()]
    )
    return metrics

def compute_mean_over_all(
        preds: list[NDArray], 
        ground_truth: list[NDArray], 
        class_map: dict[int, str],
        rois: NDArray | None = None,  # 2D
):
    """
    Averages metrics over classes
    """
    assert len(preds) == len(ground_truth)
    pos_results = []
    if rois is None:
        rois = [rois] * len(preds)
    class_results = {val: [] for val in class_map.values()} | {MEAN: []}
    for pred,truth,roi in tqdm(zip(preds, ground_truth, rois)):
        metrics = compute_full_metrics(pred, truth, roi, class_map)
        pos_results.append(metrics[POS])
        for key in class_results:
            class_results[key].append(metrics[CLS][key])
    mean_metrics = {POS: dict_average(pos_results), CLS: {}}
    for key in class_results:
        mean_metrics[CLS][key] = dict_average(class_results[key])
    return mean_metrics

def load_translation(raw: dict[str, int|None]) -> dict[int, int]:
    return {int(key): val for key,val in raw.items() if val is not None}

def translate_detections(dets: NDArray, translation_map: dict[int, int]) -> NDArray: 
    dets = deepcopy(dets)
    kept = []
    for row in dets:
        old_class = row[0]
        if old_class in translation_map:
            to_keep = row
            to_keep[0] = translation_map[old_class]
            kept.append(to_keep)
    if not kept:
        return np.ndarray((0,6))
    else:
        return np.array(kept, ndmin=2)

def eval_detection_run(
        det_folder: Path, 
        ground_truth_manifest: Path, 
        config: Path,
        out: Path, 
        split: str = TEST, 
        pattern: str = '*',
        translate: bool = False
):

    # Get run details
    run_conifg = load_config(config, EVAL)
    # translator = run_conifg[TRANS] if translate else None
    # if translator is not None:
    #     translator = load_translation(translator)
    im_res = np.array(run_conifg[RES])
    with open(ground_truth_manifest, 'r', encoding='utf-8') as f:
        manifest_data = yaml.safe_load(f)
    class_map = manifest_data[CLASS_MAP]
    label_root = Path(manifest_data[LABEL_ROOT]).resolve()
    label_files = labels_by_video(label_root / manifest_data[split], label_root)
    bbox_transform = yolo_dets_to_bbox_transform(im_res)

    out.mkdir(exist_ok=True, parents=True)
    for run_folder in det_folder.glob(pattern):

        # Bolt on fix for now
        config_dir = config.parent
        source = run_folder.stem.split('_',1)[1].rsplit('_',1)[-2]
        translator = load_config(config_dir / (source + '.json'), EVAL)[TRANS] if translate else None
        if translator is not None:
                translator = load_translation(translator)


        run_name = run_folder.stem
        run_out = out / (run_name + JSON)
        pred_folder = run_folder / LABEL_SUBFOLDER
        results = {}
        for vid in label_files:
            print(f"Processing {vid}...")
            preds = []
            truth = []
            for label_file in label_files[vid]:
                # Load paired preds and ground truth
                pred_file = pred_folder / label_file.name
                pred = load_yolo_dets(pred_file, bbox_transform, force_dim=BBOX_SIZE+2, allow_missing=True)
                label = load_yolo_dets(label_file, bbox_transform, force_dim=BBOX_SIZE+1)
                if translator is not None:
                    pred = translate_detections(pred, translator)
                preds.append(pred)
                truth.append(label)
            results[vid] = compute_mean_over_all(preds, truth, class_map=class_map)

                
        results[MEAN] = dict_average([results[key] for key in results])
        with open(run_out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)


def eval_track_performance(
        pred_start: NDArray, 
        pred_stop: NDArray, 
        label_start: NDArray, 
        label_stop: NDArray,
        threshold: float = 0.5,
    ) -> tuple[dict[str, float], bool]: # Metrics, if there was anything to match
        relevant_ids = set(label_start[:,-1]) & set(label_stop[:,-1])
        relevant_ids = list(relevant_ids)
        label_start_relevant = label_start[np.isin(label_start[:,-1], relevant_ids)]
        label_stop_relevant = label_stop[np.isin(label_stop[:,-1], relevant_ids)] 
        assert len(label_start_relevant) == len(label_stop_relevant)
        
        start_match, metrics = match_objects(pred_start, label_start_relevant, threshold, cls=None, ignore_det_class=True)
        stop_match, _ = match_objects(pred_stop, label_stop_relevant, threshold, cls=None, ignore_det_class=True)
        tp, _, fn = metrics
        total = tp + fn
        missed = fn / total if total else 0

        id_map = {pred_start[j][-1]: label_start_relevant[i][-1]  for i,j in start_match.items() if j is not None}
        tracked = len(id_map.keys())
        kept = 0
        for i,j in stop_match.items():
            if j is not None and id_map.get(pred_stop[j][-1], None) == label_stop_relevant[i][-1]:
                kept += 1

        maintained = kept / tracked if tracked else 1
        maintained_total = kept / total if total else 1

        return {
            MISSED: missed,
            # DROPPED: 1 - (missed + maintained_total),
            MAINTAINED: maintained_total,
            # DROPPED + PROPORTIONAL_SUFFIX: 1 - maintained,
            MAINTAINED + PROPORTIONAL_SUFFIX: maintained
        }, total > 0


def zoom_labels(labels: NDArray, res: NDArray, alpha: NDArray, theta: float) -> NDArray:
    zoom_scales = (res * theta)
    zoom_center = alpha * res
    zoom_box = np.concat((zoom_center - (zoom_scales), zoom_center + (zoom_scales)))
    zoom_scales *= 2 # Theta is half length
    cropped = crop_detections(labels, zoom_box)

    scale_factor = res / zoom_scales
    scale_anchor = zoom_box[:2] 
    zoomer = np.identity(5)
    zoomer[(0,1),(0,1)] = scale_factor
    zoomer[(2,3),(2,3)] = scale_factor
    zoomer[-1, :2] = -scale_factor*scale_anchor
    zoomer[-1, 2:-1] = -scale_factor*scale_anchor
    affine_bboxes = np.ones((cropped.shape[0], BBOX_SIZE + 1))
    affine_bboxes[:, :-1] = cropped[:, 1:BBOX_SIZE+1]
    scaled = np.matmul(affine_bboxes, zoomer)
    
    cropped[:, 1:BBOX_SIZE+1] = scaled[:,:-1] # Affine dim will be 1
    return cropped



label_from_image = lambda im: (im.parent / LABEL_SUBFOLDER / im.name).with_suffix(LABEL_SUFFIX) 
def eval_track_interval(interval_folder: Path, labels: list[Path], res: NDArray, translator: dict[int,int] | None, threshold: float = 0.5) -> tuple[dict[str, float], bool]:
    zoom_file = interval_folder / ZOOM_DATA
    zoomed = zoom_file.is_file()
    images = sorted([file for file in interval_folder.glob('*.*') if file not in (zoom_file,)])
    start_pred = label_from_image(images[0])
    stop_pred = label_from_image(images[-1])

    # Find matching ground truth
    start_label = None
    stop_label = None
    for label in labels:
        name = label.name
        if name == start_pred.name:
            start_label = label
            if stop_label is not None:
                break
        if name == stop_pred.name:
            stop_label = label
            if start_label is not None:
                break
    assert start_label is not None and stop_label is not None

    # Load data
    transform = yolo_dets_to_bbox_transform(res)
    start_pred = load_yolo_dets(start_pred, transform, force_dim= BBOX_SIZE + 3, allow_missing=True)
    stop_pred = load_yolo_dets(stop_pred, transform, force_dim= BBOX_SIZE + 3, allow_missing=True)
    start_label = load_yolo_dets(start_label, transform, force_dim= BBOX_SIZE + 2)
    stop_label = load_yolo_dets(stop_label, transform, force_dim= BBOX_SIZE + 2)
    if zoomed:
        with open(zoom_file, 'r', encoding='utf-8') as f:
            zoom_data = json.load(f)
        stop_label = zoom_labels(stop_label, res, np.array(zoom_data[ALPHA]), zoom_data[THETA])
    if translator is not None:
        start_pred = translate_detections(start_pred, translator)
        stop_pred = translate_detections(stop_pred, translator)

    return eval_track_performance(start_pred, stop_pred, start_label, stop_label, threshold)

def eval_vid_intervals(
        vid_folder: Path, 
        labels: list[Path], 
        res: NDArray, 
        translator: dict[int, int] | None,
        threshold: float = 0.5, 
        include_blank: bool = False
) -> dict[str, float] | None:
    results = []
    for interval in tqdm(vid_folder.glob('*')):
        interval_results, valid = eval_track_interval(interval, labels, res, translator, threshold)
        if valid or include_blank:
            results.append(interval_results)
    return dict_average(results) if results else None
        
TOGGLE_DISPLAY = {True: WITH, False: WITHOUT}
def eval_tracker_run(        
    run_folder: Path, 
    ground_truth_manifest: Path, 
    config: Path,
    out: Path, 
    split: str = TEST, 
    pattern: str = '*',
    translate: bool = False,
) -> Path:

    # Get run details
    run_conifg = load_config(config, EVAL)
    run_conifg |= load_config(config, TRACK)
    translator = run_conifg[TRANS] if translate else None
    if translator is not None:
        translator = load_translation(translator)
    target_res = np.array(run_conifg[TARGET_RES])
    with open(ground_truth_manifest, 'r', encoding='utf-8') as f:
        manifest_data = yaml.safe_load(f)
    label_root = Path(manifest_data[LABEL_ROOT]).resolve()
    label_files = labels_by_video(label_root / manifest_data[split], label_root, tracks=True)

    for tracker in run_folder.glob(pattern):
        results_out = out / (tracker.stem + JSON)
        results = {}
        for vid_folder in tracker.glob('*'):
            vid_name = vid_folder.stem
            results[vid_name] = {} 
            for blank_toggle in (True, False):
                res = eval_vid_intervals(
                    vid_folder, label_files[vid_name], target_res, translator, include_blank=blank_toggle
                )
                results[vid_name][f"{TOGGLE_DISPLAY[blank_toggle]}_{BLANK}"] = res if res is not  None else {}
            
        results[MEAN] = dict_average([val for key,val in results.items() if val is not None])
        with open(results_out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

    return results_out



    
    

if __name__ == "__main__":
    # CLI to train detectors
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Run detection metrics.')
    parser.add_argument('--detections', '-d', type=str, required=True, help='Folder of output labels')
    parser.add_argument('--manifest', '-m', type=str, required=True, help='Manifest for groundtruth dataset.')
    parser.add_argument('--split', '-s', type=str, default=TEST, help='split used for evaluation')
    parser.add_argument('--config', '-c', type=str, required=True, help='Config for resolution and optional class map')
    parser.add_argument('--out', '-o', type=str, required=True, help='Path to output json.')
    parser.add_argument('--pattern', '-p', type=str, default='*', help='glob pattern for run folders to use')
    parser.add_argument('--tracking', '-t', action='store_true', help='Eval Tracking experiment')
    parser.set_defaults(tracking=False)
    parser.add_argument('--translate', '-r', action='store_true', help='Translate detection classes according to config')
    parser.set_defaults(translate=False)
    args = parser.parse_args()

    config = Path(args.config).resolve()
    det_folder = Path(args.detections).resolve()
    manifest = Path(args.manifest).resolve()
    out = Path(args.out).resolve()
    split = args.split
    tracking = args.tracking
    pattern = args.pattern
    translate = args.translate

    if tracking:
        eval_tracker_run(det_folder, manifest, config, out, split, pattern, translate)
    else:
        eval_detection_run(det_folder, manifest, config, out, split, pattern, translate)

        
        
        
    