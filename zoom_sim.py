import cv2
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
from typing import Iterator, Iterable
from pathlib import Path
from typing import Callable

from dji_video import TimedImage, dji_interval_generator, FILE_TIMESTAMP_FORMAT
from utils import labels_by_video
from utils import load_yolo_dets

ZoomRegion = tuple[tuple[float, float], float]


# Optimised procedure for a future paper that I am butchering to quickly mimic some behavior
IPRO_BOUND = 0.11 / 2 # I-Pro camera hardware limit.
def select_ROI(
        points: np.ndarray, weights: np.ndarray, bc: float = IPRO_BOUND, k: float = 3
) -> ZoomRegion:
    """
    Determine the region to zoom into 

    Args:
        points (NDArray): Array of points, normalised to unit square
        weights (NDArray): Point weights, index must correspond to point.

    Returns:
        tuple[NDArray[float, float], float]: center of region, radius of region.
    """
    # points is an array (Nx2), so is weights (N) (O(1) Access),
    N = len(points)
    K = k * np.sum(weights) # TODO UPDATE SUPP MATERIAL and PAPER EQ TODO UPDATE K NAMING
    I = range(N)
    neighborhood_distances = np.zeros(N) # Stores distances for sorting

    best_best_obj = float("inf") # Best objective 
    best_best_alpha, best_best_theta = 0.5 * np.ones(2), bc # Best region
    for i in I: # Start points O(N)
        best_obj = float("inf") # Best objective within loop 
        best_alpha, best_theta = 0.5 * np.ones(2), bc # Best region within loop 

        # Determine iteration order
        for j in I: # O(N)
            neighborhood_distances[j] = np.linalg.norm(points[i] - points[j]) #L2 norm
        neighborhood = np.argsort(neighborhood_distances) # O(Nlog(N))
        
        triangle = np.zeros(3, dtype=np.uint8) # Furthest point indexes, O(1) Access
        triangle_distances = np.zeros(3) # Distances of furthest points, O(1) Again
        centroid = np.zeros(2) # Current estimate of center
        centroid_score = 0 # current weighted sum of distances within region
        t2 = sum(weights) # Current Second term O(N)
        w = 0 # Sum of weights on past iteration
        for q,j in enumerate(neighborhood): # O(N)
            new_centroid = ((w*centroid) + (weights[j] * points[j])) / (w + weights[j]) # O(1)
            centroid_score =  centroid_score + (w * np.linalg.norm(new_centroid - centroid)**2) + (weights[j] * np.linalg.norm(new_centroid - points[j])**2)
            centroid = new_centroid

            t2 = t2 - weights[j] # O(1)
            w = w + weights[j]

            # new theta in O(3)
            new_distance = np.linalg.norm(centroid - points[j])
            if q < 3:
                triangle[q] = j
                triangle_distances[q] = new_distance
                for k in range(q):
                    triangle_distances[k] = np.linalg.norm(centroid - points[triangle[k]])
                triangle_hierarchy = np.argsort(triangle_distances[0:q+1])

                theta = triangle_distances[triangle_hierarchy[-1]]
            else:
                for k in range(3):
                    triangle_distances[k] = np.linalg.norm(centroid - points[triangle[k]])
                triangle_hierarchy = np.argsort(triangle_distances) # O(3log(3)) < O(2*3)

                triangle_closest = triangle_distances[triangle_hierarchy[0]]
                triangle_furthest = triangle_distances[triangle_hierarchy[-1]]
                if new_distance >= triangle_closest:
                    # Shift triangle outward
                    triangle[triangle_hierarchy[0]] = j
                    triangle_distances[triangle_hierarchy[0]] = new_distance
                theta = max(new_distance, triangle_furthest, bc)

            # Determine if objective has improved
            objective = centroid_score + (2*t2) + (K * theta**2)
            if objective < best_obj:
                best_obj = objective
                best_alpha, best_theta = centroid, theta
            
        
        # Determine if this neighborhood yielded better results
        if best_obj < best_best_obj:
            best_best_obj = best_obj
            best_best_alpha, best_best_theta = best_alpha, best_theta

    return best_best_alpha, best_best_theta









def compute_bbox(
        alpha: NDArray, 
        theta: float, 
        res: NDArray
) -> NDArray: # [x1,y1,x2,y2]
    """
    Helper to compute the bounding box for zoom operation
    """
    # Exploiting numpy parallelism
    bbox = np.rint(np.concatenate((((alpha - theta) * res), ((alpha + theta) * res))))
    bbox[0::2] = bbox[0::2].clip(min=0, max=res[0])
    bbox[1::2] = bbox[1::2].clip(min=0, max=res[1])
    return bbox

# new view? copy?
def zoom_frame(frame: NDArray, og_res: NDArray, target_res: NDArray, alpha: NDArray, theta: float) -> NDArray:
    zoom_box = tuple(map(int, compute_bbox(np.array(alpha), theta, og_res)))
    crop = frame[zoom_box[1]:zoom_box[3], zoom_box[0]:zoom_box[2]] 
    return cv2.resize(crop, target_res, interpolation = cv2.INTER_AREA)

# TODO test this iterator
def simulated_zoom_iterator(
        vid: Path, 
        keytimes: list[datetime], 
        original_size: int, 
        keep: int | tuple[int, int],
        target_res: tuple[int, int],
        zoom_regions: Iterable[ZoomRegion] | None = None, # Will default to center max zoom
        overlap: bool = True,
        return_region: bool = False,
        label_file: Path | None = None,
) -> Iterator[tuple[TimedImage]] | Iterator[tuple[TimedImage], ZoomRegion]:
    # Process output shape
    if isinstance(keep, int):
        keep = (keep, keep)
    left, right = keep
    assert left and right and left + right <= original_size

    # Casting for parrallelism
    target_res = np.array(target_res)

    labels = None
    if label_file is not None:
        labels = labels_by_video(label_file)[vid.stem]

    region_index = 0
    for interval in dji_interval_generator(vid, keytimes, original_size, overlap):
        # Determine zoom
        og_res = np.array(interval[0][1].shape[-2::-1]) # Interval resolutions should be homogeneous
        zoom_limit = max(target_res/og_res) / 2
        if zoom_regions is not None: 
            zoom_region = zoom_regions[region_index]
            region_index += 1
            if region_index >= len(zoom_regions):
                region_index = 0
        else:
            if labels is not None:
                # Grab labels corresponding to final keytime
                target = datetime.strftime(interval[-1][0], FILE_TIMESTAMP_FORMAT)
                label = None
                for path in labels:
                    if path.stem == target:
                        label = path
                        break
                if label == None:
                    print(f"Warning: Label missing for {target}, zooming into center.")
                    zoom_region = ((0.5,0.5), zoom_limit)
                else:
                    # Compute minimial zoom box with most boxes
                    dets = load_yolo_dets(label)[:,1:3]
                    zoom_region = select_ROI( # Big penalty for box size forces smallest
                        dets, np.ones(dets.shape[0]), zoom_limit, 10
                    )
                    zoom_region = tuple(zoom_region[0]), zoom_region[1] # Buh

            else:
                # Fallback to just center zoom
                zoom_region = ((0.5,0.5), zoom_limit)

            
        alpha = np.array(zoom_region[0])
        theta = zoom_region[-1]

        zoomed_interval = [
            (timed_frame[0], cv2.resize(timed_frame[1], target_res, interpolation = cv2.INTER_AREA)) 
            for timed_frame in interval[:left]
        ] + [
            (timed_frame[0], zoom_frame(timed_frame[1], og_res, target_res, alpha, theta)) 
            for timed_frame in interval[-right:]
        ]

        yield zoomed_interval, zoom_region if return_region else zoomed_interval


def full_tour(zoom_limit: float, index: int) -> tuple[tuple[float, float], float]:
    tour_side_len = 1 // (2*zoom_limit)
    theta = 0.5 / tour_side_len
    i = index % (tour_side_len**2) # Wrap to start
    y, x = i // tour_side_len, i % tour_side_len
    return ((((2*x)+1) * theta, ((2*y)+1) * theta), theta)

# TODO clean up
def count_detections(dets: NDArray, tour_location: tuple[tuple[float,float], float], filter: Callable[[NDArray], NDArray] | None = None) -> int:
    alpha, theta = tour_location
    center_x, center_y = alpha
    
    count = 0
    for box in dets:
        box_x, box_y = box[1:3]
        if (abs(box_x - center_x) < (theta)  and abs(box_y - center_y) < (theta)) and (filter is None or filter(box[np.newaxis,:])):
            count += 1
    return count

# TODO this (and others?) need to be preserved in final codebase
def compute_standard_tours(
        image_file: Path, 
        tour_length: int = 5, 
        img_res: tuple[int, int] = (3840, 2160), 
        target_res: tuple[int, int] = (640, 480), 
        filter: Callable[[NDArray], NDArray] | None = None
) -> dict[str, list[ZoomRegion]]:    
    # Partition unit square into minimal tour regions
    zoom_limit = max(np.array(target_res)/np.array(img_res)) / 2
    zones = []
    i = 0
    candidate = full_tour(zoom_limit, i) # Quick and dirty reuse of code
    while candidate not in zones:
        zones.append(candidate)
        i+=1
        candidate = full_tour(zoom_limit, i)
    assert len(zones) >= tour_length # prevent infinite loop

    root = image_file.parent
    with open(image_file, 'r', encoding='utf-8') as f: # TODO constants
        label_files = [root / line.rstrip().replace('images','labels').replace('.png','.txt') for line in f.readlines()]

    label_counts = {}
    for label_file in label_files:
        vid = label_file.parent.stem
        if vid not in label_counts:
            label_counts[vid] = {zone: 0 for zone in zones}
        
        # Add current boxes to count
        boxes = np.genfromtxt(label_file, delimiter=' ', ndmin=2)
        for zone in label_counts[vid]:
            label_counts[vid][zone] += count_detections(boxes, zone, filter)

    # Determine maximum regions
    tours = {}
    for vid in label_counts:
        # Determine maximum regions
        standard_tour = []
        while len(standard_tour) < tour_length:
            max_count = -1
            max_zone = None
            for zone, count in label_counts[vid].items():
                if count > max_count:
                    max_count = count
                    max_zone = zone
            assert max_zone is not None

            standard_tour.append(max_zone)
            label_counts[vid].pop(max_zone)

        tours[vid] = standard_tour   
    return tours


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Compute standard tours for simulated zoom.')
    parser.add_argument('--image_file', '-i', type=str, required=True, help='Path to file containing list of frames to use.')
    parser.add_argument('--config', '-c', type=str, required=True, help='Config containing original resolution information')
    parser.add_argument('--out', '-o', type=str, required=True, help='Path to output json.')
    parser.add_argument('--res', '-r', type=int, nargs=2, default=(640, 480), help='Target zoom resolution')
    parser.add_argument('--num', '-n', type=int, default=5, help='Tour size')
    args = parser.parse_args()

    from utils import load_config
    import json

    image_list = Path(args.image_file).resolve()
    config = Path(args.config).resolve()
    out = Path(args.out).resolve()
    target = args.res
    num = args.num

    config = load_config(config, 'eval')
    og_res = config['res']

    tours = compute_standard_tours(image_list, num, og_res, target)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(tours, f, indent=4)
