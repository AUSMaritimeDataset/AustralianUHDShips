from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
from cv2 import findHomography, RANSAC
import json
from copy import deepcopy

# That documentation did not lie; Shit is modular
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantAcceleration # ,ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.dataassociator.probability import JPDAwithLBP
from stonesoup.dataassociator.neighbour import NearestNeighbour, GNNWith2DAssignment, DataAssociator
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, Updater
from stonesoup.measures import SquaredMahalanobis, Euclidean
from stonesoup.functions import gm_reduce_single

from eval_tracker import TimedImage, track_interval, get_params, TEST, TRACK, \
    DATASET_NAME, TARGET_RES, load_model, get_keytimes, simulated_zoom_iterator, \
    VIDEO_EXT, FILE_TIMESTAMP_FORMAT, ZOOM_DATA, ZoomRegion, dji_interval_generator
from metrics import label_from_image, LABEL_SUBFOLDER, LABEL_SUFFIX
from utils import load_yolo_dets


TimedDetection = TimedImage # Same structure different content

# TODO parameterise
INTERVAL_SIZE = 20
ZOOM_KEEP = 9
OVERLAP = True

# TODO fix or parameterise
DROP_TRACKS = True
DISCOVER_TRACKS = True
TENTATIVE_SURVIVE = True

DET_SIZE = 4 + 3 # class, bbox, conf, id

# Fixed tracking parameters
NDIM = 2
NDERIV = 3
PROB_GATE = 0.95 # Probability that detection of an object will reflect its true state (in our case detection is position so basically certain)
MAX_MISSES = 3
MAX_TENTATIVE_MISSES = 2
REQ_HITS = 4
NN_DISTANCE = 5.991 # 95% 13.82 # 99.9% Confidence interval (Chisquared 2DOF)


RANSAC_ITER_CAP = 1024
RANSAC_DIST = 0.005
ZOOM_ASSOC_DIST = 0.05 # L2 dist, Mahalobis model breaks over zoom

# Infrastructure
DET_INDEX = 'index'
REFERENCE_METHOD = 'ref_method'

def init_track(
    detection: Detection,
    pos_sigma: float,
    vel_sigma: float,
    acc_sigma: float,
) -> Track:
    """
    Initialise new tentative track. 

    Args:
        detection (Detection): Detection inciting new track.
        pos_sigma (float): Uncertainty in position (hyperparameter)
        vel_sigma (float): Uncertainty in velocity (hyperparameter)
        acc_sigma (float): Uncertainty in acceleration (hyperparameter)

    Returns:
        Track: New Tentative Track
    """
    state_v = np.zeros(NDERIV * NDIM)
    state_v[0::NDERIV] = detection.state_vector.flatten() # Fucky infill

    state = GaussianState(
            StateVector(state_v),
            np.diag([pos_sigma**2,
                    vel_sigma**2,
                    acc_sigma**2] * NDIM),
            timestamp=detection.timestamp,
        )
    state.measurement = detection
    track = Track([state])
    # We bolt our own properties to the side to track (haha) viability (War crime)
    track.hits = 1 
    track.missed = 0

    return track

def update_gaussian(
        track: Track,
        hypothesis: SingleHypothesis,
        updater: KalmanUpdater
) -> Detection:
    """
    Updates Tentative Hypotheses based on most likely greedy hypothesis.

    Args:
        track (Track): Tentative Track
        hypothesis (SingleHypothesis): Most likely Hypothesis
        updater (KalmanUpdater): Updater to compute posterior

    Returns:
        Detection: Detection absorbed by track.
    """
    if hypothesis: # Falsy if missed detection
        update = updater.update(hypothesis)
        update.measurement = hypothesis.measurement # Bolting a flattened interface for my specific nefarious purposes                
    else:
        update = hypothesis.prediction
        update.measurement = MissedDetection(timestamp=update.timestamp) # The main reason for flattened interface
            
    track.append(update)
    return update.measurement

def update_gaussian_mixture(
        track: Track, 
        hypotheses: MultipleHypothesis, 
        updater: KalmanUpdater
) -> Detection: 
    """
    Updates confirmed track using JPDA Multiple hypothesis.

    Args:
        track (Track): Track to update
        hypotheses (MultipleHypothesis): Most likely JPDA hypothesis.
        updater (KalmanUpdater): Updater to compute posterior.

    Returns:
        Detection: Detection absorbed by track.
    """
    AV = False
    if AV:
        components = []
        weights = []
        max_weight = 0
        sample_det = None
        for h in hypotheses:
            # Track most likely point
            prob = h.probability
            weights.append(prob)
            if prob > max_weight:
                max_weight = prob
                sample_det = h.measurement
            
            # Update mixture
            if h:
                components.append(updater.update(h))
            else:
                components.append(h.prediction)
        
        # Reduce mixture of states to one posterior estimate Gaussian.
        means = StateVectors([state.state_vector for state in components])
        covars = np.stack([state.covar for state in components], axis=2)
        weights = np.asarray(weights) / sum(weights) # JPDA not neccessarily normalised
        post_mean, post_covar = gm_reduce_single(means, covars, weights)
    
    else:
        # Extract best component (Sparsity meant above lead to floating)
        max_weight = 0
        sample_det = None
        component = None
        for h in hypotheses:
            prob = h.probability
            if prob > max_weight:
                max_weight = prob
                sample_det = h.measurement
                if h:
                    component = updater.update(h)
                else:
                    component = h.prediction
        post_mean = component.state_vector
        post_covar = component.covar

    # Update Track
    hypotheses.measurement = sample_det # Here I go bolting again (This is a cry for help)
    new_state = GaussianStateUpdate(
            post_mean, post_covar,
            hypotheses,
            sample_det.timestamp,
    )
    new_state.measurement = sample_det # WHEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    track.append(new_state)
    return sample_det

def clamp_velocities(tracks: set[Track], threshold: float, forward: bool = True, ignore_acc: bool = False): # In place
    for track in tracks:
        state_vector = track[-1 if forward else 0].state_vector
        vel = state_vector[1::NDERIV,0]
        acc = state_vector[2::NDERIV,0]
        if np.linalg.norm(vel) < threshold and (ignore_acc or np.linalg.norm(acc) < threshold): # L2 norm
            state_vector[1::NDERIV,0] = 0 # statevector forces column rep
            state_vector[2::NDERIV,0] = 0
            # If I had realised this sooner I would have saved a lot of trouble


CONRER_TRANSFORM = np.c_[np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
]), np.ones(4)].T

SAMPLE_MIN = 4
# TODO PARAMETERISE
MIN_INLIERS = SAMPLE_MIN # Leaving this open for later tuning
HOMO_VALIDATE_PARAMS = {
'min_inlier_ratio': 0.6,
'max_corner_displacement': 0.10,
'min_area_ratio': 0.8,
'max_area_ratio': 1.2,
'max_anisotropy': 1.25,
'max_perspective': 0.05
}

def validate_homography(
        H: NDArray,
        mask:  NDArray,
        min_inlier_ratio: float =0.6,
        max_corner_displacement: float =0.15,
        min_area_ratio: float  =0.7,
        max_area_ratio: float =1.3,
        max_anisotropy:  float =1.5,
        max_perspective:float =0.1,
) -> bool:
    if H is None or mask is None:
        return False

    if not np.all(np.isfinite(H)):
        return False

    # Normalise H so thresholds are meaningful
    if abs(H[2, 2]) < 1e-10:
        return False
    H = H / H[2, 2]

    # 1. RANSAC support
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < MIN_INLIERS:
        return False

    if inliers.mean() < min_inlier_ratio:
        return False

    # 2. Reject excessive projective distortion
    # OpenCV homography convention:
    # [h00 h01 h02]
    # [h10 h11 h12]
    # [h20 h21 h22]
    if np.linalg.norm(H[2, :2]) > max_perspective:
        return False

    # Transform corners of unit square
    corners_h = np.c_[CONRER_TRANSFORM, ]
    transformed_h = (H @ corners_h.T).T

    # Don't permit denominator approaching zero or changing sign
    w = transformed_h[:, 2]
    if np.any(np.abs(w) < 0.25):
        return False

    transformed = transformed_h[:, :2] / w[:, None]

    # No giant global displacement
    displacement = np.linalg.norm(transformed - CONRER_TRANSFORM[:-1,:].T, axis=1)
    if displacement.max() > max_corner_displacement:
        return False

    # Polygon area
    def polygon_area(p):
        x = p[:, 0]
        y = p[:, 1]
        return 0.5 * abs(
            np.dot(x, np.roll(y, 1))
            - np.dot(y, np.roll(x, 1))
        )

    area_ratio = polygon_area(transformed)  # original unit square area = 1

    if not (min_area_ratio <= area_ratio <= max_area_ratio):
        return False

    # 3. Check local scaling / squash near image centre
    x, y = 0.5, 0.5

    a, b, c = H[0]
    d, e, f = H[1]
    g, h, _ = H[2]

    den = g*x + h*y + 1.0
    nx = a*x + b*y + c
    ny = d*x + e*y + f

    J = np.array([
        [
            (a*den - g*nx) / den**2,
            (b*den - h*nx) / den**2,
        ],
        [
            (d*den - g*ny) / den**2,
            (e*den - h*ny) / den**2,
        ]
    ])

    s = np.linalg.svd(J, compute_uv=False)

    if s[-1] <= 1e-8:
        return False

    # Biggest local scale / smallest local scale
    anisotropy = s[0] / s[-1]

    if anisotropy > max_anisotropy:
        return False

    return True


def compute_correction( # TODO note returns transposed
        confirmed_tracks: set[Track], 
        zoom_dets: set[Detection], 
        zoom_time: datetime,
        # predictor: KalmanPredictor,
        associator: DataAssociator,
        # neighbors: int = 3,
        # max_neighbor_dist: float = 0.15,
        max_ransac_iters: int = 1024,
        ransac_theshold: float = 0.01,
        validate: bool = True
        # normalise: bool = True,
): #TODO copy docstrings from commented code blocks   
    good_luck = lambda: np.eye(NDIM + 1)
    if len(confirmed_tracks) < SAMPLE_MIN or len(zoom_dets) < SAMPLE_MIN: # Good luck
        return good_luck()

    ordered_dets = list(zoom_dets)
    # det_states = np.array([det.state_vector.flatten() for det in ordered_dets], ndmin=2)  # (M,2)

    # Get K nearest neighbor associations for RANSAC # TODO may need to make K 1
    
    # Determine associated points
    associations = associator.associate(
        confirmed_tracks, zoom_dets, zoom_time
    )
    src_points = []
    dst_points = []
    # matched_detections = []
    # match_distances = []
    for track, hypothesis in associations.items():
        if not hypothesis: # Missed det is falsy
            continue

        det = np.asarray(
            hypothesis.measurement.state_vector,
            dtype=float,
        ).reshape(-1)

        pred = np.asarray(
            hypothesis.prediction.state_vector,
            dtype=float,
        ).reshape(-1)

        src_points.append(det) # Observation is just pos
        dst_points.append(pred[::NDERIV]) # State is interwoven
        # matched_detections.append(hypothesis.measurement)
        # match_distances.append(float(hypothesis.distance))

    if len(src_points) < SAMPLE_MIN:
        return good_luck()

    H, mask = findHomography(np.array(src_points), np.array(dst_points), method=RANSAC, maxIters=max_ransac_iters, ransacReprojThreshold=ransac_theshold)
    
    #Validate homography is practical
    return H.T if not validate or validate_homography(
        H,
        mask,
        **HOMO_VALIDATE_PARAMS
    ) else good_luck()

def apply_correction(detections: set[Detection], correction: NDArray) -> set[Detection]:
    ordered_dets = list(detections)
    if not len(ordered_dets):
        return detections # Nothing to do
    
    # Construct homogenised monololith for efficient parralellised computation
    the_goober = np.ones((len(ordered_dets), 3))
    the_goober[:,:NDIM] = np.array([det.state_vector.flatten() for det in ordered_dets], ndmin=2)

    corrected = np.matmul(the_goober, correction) # Numpy hierarchy is transposed, so H should be 
    for det, correct in zip(ordered_dets, corrected[:,:NDIM] / corrected[:,NDIM,np.newaxis]):
        det.state_vector = StateVector(correct)
    
    return set(ordered_dets)

# TODO reorder functions
def translate_to_detsets(
    interval: list[TimedDetection], 
    jump_frame: int, # Index,
    alpha: NDArray, 
    theta: float,
    fov: NDArray | None = None,
) -> list[datetime, set[Detection]]:
    # Compute unzoom operation
    if fov is not None:
        raise NotImplementedError("Lens effects to come later")
    else:
        # Affine (Digital) Zoom
        the_unzoomer = np.diag(([2*theta]*(2*NDIM))+[1])
        the_unzoomer[-1, 0:NDIM] = alpha - theta # It is NOT interwoven here dumbass
        unzoom_op = lambda mat: np.matmul(
                np.c_[mat, np.ones(mat.shape[0])], the_unzoomer # numpy heirarchy transposed
        )[:,:-1] # Strip affine dim (will be 1)

    # Translate to stonesoup format
    detsets = []
    for i, val in enumerate(interval):
        time, dets = val
        dets = deepcopy(dets) # This will be slow but required
        if i >= jump_frame:
            dets[:,1:(2*NDIM)+1] = unzoom_op(dets[:,1:(2*NDIM)+1])
    
        detections = set()
        for j,det in enumerate(dets):
            detections.add(
                Detection(
                    det[1:NDIM+1], # Only pos is tracked
                    timestamp = time,
                    metadata={
                        DET_INDEX: j # For associate back to OG detection structure
                    }
                )
            )

        detsets.append((time, detections))

    return detsets


def update_tracks(
        confirmed: set[Track], 
        tentative: set[Track], 
        dets: set[Detection], 
        time: datetime,
        register: NDArray, # For easily going back to ultralytics format
        track_count: int,
        confirmed_associator: DataAssociator, 
        tentative_associator: DataAssociator,
        default_updater: Updater, # Will be overridden if detection has one
        initial_pos_sigma: float,
        initial_vel_sigma: float,
        initial_acc_sigma: float,
) -> int: # updates track sets and register in place, returns new track count
    consumed_dets = set()
    
    # Run JPDA on confirmed tracks
    association = confirmed_associator.associate(
        confirmed, dets, time
    )

    # Update confirmed tracks
    tracks_to_remove = set()
    for track, hypotheses in association.items():
        update_func = update_gaussian_mixture if isinstance(hypotheses, MultipleHypothesis) else update_gaussian
        best_det = update_func(track, hypotheses, default_updater)
        if best_det: # MissedDetection is Falsy
            consumed_dets.add(best_det)    
            track.missed = 0
            register[best_det.metadata[DET_INDEX], -1] = track.id # Record track to ultralytics dets
        else:
            track.missed += 1
            if track.missed >= MAX_MISSES:
                tracks_to_remove.add(track)

    # Zucc dropped tracks
    if DROP_TRACKS:
        for track in tracks_to_remove:
            confirmed.remove(track)

    
    if DISCOVER_TRACKS:
        # Locally assign tentative tracks
        tentative_association = tentative_associator.associate(
            tentative, dets - consumed_dets, time) # Set subtraction    

        # Confirm tentative tracks
        tracks_to_remove = set()
        for track, hypothesis in tentative_association.items():
            update_func = update_gaussian_mixture if isinstance(hypothesis, MultipleHypothesis) else update_gaussian
            best_det = update_func(track, hypothesis, default_updater)
            if best_det:
                consumed_dets.add(best_det)
                track.missed = 0
                track.hits += 1
                register[best_det.metadata[DET_INDEX], -1] = track.id # Record track to ultralytics dets
                if track.hits >= REQ_HITS:
                    # Promote track to confirmed
                    tracks_to_remove.add(track)
                    confirmed.add(track)     
            else:
                # Drop missed tracks Quickly
                track.missed += 1
                if track.missed > MAX_TENTATIVE_MISSES:
                    tracks_to_remove.add(track)
                
        # Zucc tracks that are no longer tentative
        for track in tracks_to_remove:
            tentative.remove(track)

    
        # Add any remaining datapoints as tentative new tracks
        for det in dets - consumed_dets:
            new_track = init_track(det, initial_pos_sigma, initial_vel_sigma, initial_acc_sigma)
            new_track.id = track_count
            track_count += 1
            register[det.metadata[DET_INDEX], -1] = new_track.id # Record track to ultralytics dets
            tentative.add(new_track)

    return track_count
    


def track_over_zoom_interval(
        interval: list[TimedDetection],
        jump_frame: int, # Index of first zoomed frame
        zoom_region: ZoomRegion,
        detection_prob: float = 0.75,
        motion_covar_falloff: float = 0.005, # Essentially how much velocity can change
        measurement_sigma: float = 0.1,
        initial_pos_sigma: float = 0.05,
        initial_vel_sigma: float = 0.05,
        initial_acc_sigma: float = 0.05,
        noise_density: float = 0.1,
        clamp_vel: float | None = 0.001,
        wobble_correction: bool = True,
) -> set[Track]: # Adds track ids in place

    # Get into stonesoup format
    alpha, theta = zoom_region
    alpha = np.array(alpha) # Enable parallelism
    register = [item[1] for item in interval] # for writing final tracks
    detsets = translate_to_detsets(interval, jump_frame, alpha, theta)

    # Set up (a lot of) model
    noise_covar = np.diag([measurement_sigma**2] * NDIM)
    measurement_model = LinearGaussian(
            ndim_state=NDERIV*NDIM,
            mapping=(tuple(i*NDERIV for i in range(NDIM))), # enumerate(mapping) -> row,col of matrix H to be 1 and not 0.
            noise_covar=noise_covar
    )
    zoomed_measurement_model = LinearGaussian(
            ndim_state=NDERIV*NDIM,
            mapping=(tuple(i*NDERIV for i in range(NDIM))), 
            noise_covar=noise_covar * ((2*theta)**2) # Zooming scales down uncertainty respectively
    ) 
    transition_model = CombinedLinearGaussianTransitionModel([ # Assuming roughly constant motion for N frames
            ConstantAcceleration(motion_covar_falloff), ConstantAcceleration(motion_covar_falloff),
    ])
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model) # >> Meas model overridden if measurement has one.

    # Statistical models for track assignment
    hypothesiser = PDAHypothesiser(
            predictor,
            updater,
            clutter_spatial_density=noise_density, 
            prob_detect=detection_prob, 
            prob_gate=PROB_GATE 
    )
    tentative_hypothesiser = DistanceHypothesiser(
        predictor,
        updater,
        measure=SquaredMahalanobis(),
        missed_distance=NN_DISTANCE
    ) 
    zoom_hypothesiser = DistanceHypothesiser(
        predictor,
        updater,
        measure=Euclidean(),
        missed_distance=ZOOM_ASSOC_DIST
    )
    
    track_associator = JPDAwithLBP(hypothesiser)
    tentative_associator = NearestNeighbour(tentative_hypothesiser)
    zoom_associator = GNNWith2DAssignment(zoom_hypothesiser)

    confirmed_tracks = set()
    tentative_tracks = set()
    all_tracks = set()
    track_count = 0
    correction = None
    for i, item in enumerate(zip(detsets, register)):
        frame_dets, frame_register = item
        time, dets = frame_dets

        if i == jump_frame:
            # Zoom frame shitfuckery
            if not TENTATIVE_SURVIVE:
                tentative_tracks.clear()
            if clamp_vel is not None:
                clamp_velocities(confirmed_tracks, clamp_vel) # Don't want jitter excaserbated or however you spell it
                clamp_velocities(tentative_tracks, clamp_vel)

            # Compute the fuckeningTM.
            correction = compute_correction(
                confirmed_tracks if wobble_correction else set(), # Will immediately return identity if not
                dets,
                time,
                zoom_associator,
                max_ransac_iters = RANSAC_ITER_CAP,
                ransac_theshold=RANSAC_DIST
            )

        if i >= jump_frame:
            assert correction is not None
            # Apply corrections before association
            dets = apply_correction(dets, correction)
            for det in dets: # Detections are unzoomed
                det.measurement_model = zoomed_measurement_model
            
        track_count = update_tracks(
            confirmed_tracks,
            tentative_tracks,
            dets,
            time,
            frame_register,
            track_count,
            track_associator if i != jump_frame else tentative_associator, # JPDA assumptions violated
            tentative_associator,
            updater,
            initial_pos_sigma,
            initial_vel_sigma,
            initial_acc_sigma,
        )
        all_tracks |= tentative_tracks

    return all_tracks
        

def load_timed_detections(path: Path) -> TimedDetection:
    return datetime.strptime(path.stem, FILE_TIMESTAMP_FORMAT), \
        load_yolo_dets(path, force_dim=DET_SIZE, allow_missing=True)

def save_timed_detections(dets: TimedDetection, out_dir: Path):
    fmt = ['%d'] + ['%0.5f'] * (dets[1].shape[-1] -1)
    if len(fmt) >= DET_SIZE:
        fmt[-1] = fmt[0]
    np.savetxt(
        (out_dir / datetime.strftime(dets[0], FILE_TIMESTAMP_FORMAT)).with_suffix(LABEL_SUFFIX),
        dets[1], delimiter=' ', fmt=fmt
    )

# Loads timed detections without tracks to benchmark our method
def strip_tracks(results_folder: Path) -> list[TimedDetection]:
    dets = [load_timed_detections(det) for det in [label_from_image(im) for im in sorted([im for im in results_folder.glob('*.*') if im.name not in (ZOOM_DATA,)])]]
    for _, det in dets:
        det[:,-1] = -1 # Strip track id
    return dets


DEFAULT_TRACKER = 'botsort.yaml'
def run_zoom_tracking(
        checkpoints: list[Path],
        image_file: Path,
        video_folder: Path,
        config: Path,
        out: Path,
        device: int | str,
        zooming: bool = True,
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
    target_res = track_params.pop(TARGET_RES)
    keytimes = get_keytimes(image_file)
    if zoom_tours is not None:
        with open(zoom_tours, 'r', encoding='utf-8') as f:
            zoom_tours = json.load(f)
    reference_params = get_params(config, mode=REFERENCE_METHOD, **kwargs)

    og_interval_size = INTERVAL_SIZE
    kept = ZOOM_KEEP

                
    for checkpoint in checkpoints:
        # Set up output
        run_name = checkpoint.stem
        if dataset_name:
            run_name += f"_{dataset_name}"
        run_dir = out / run_name
    
        model = load_model(checkpoint)

        for vid in keytimes:
            # Set up video output
            vid_out = run_dir / vid
            vid_out.mkdir(parents=True, exist_ok=False)

            if zooming:
                tour = zoom_tours[vid] if zoom_tours is not None else zoom_tours
                for i, zoom_interval in enumerate(simulated_zoom_iterator(video_folder / (vid + VIDEO_EXT),
                                                    keytimes[vid], og_interval_size, kept, target_res,
                                                    overlap=OVERLAP, return_region=True, zoom_regions=tour, 
                                                    label_file=image_file)):
                    interval, region = zoom_interval
                    working_dir = track_interval(
                                    model,
                                    DEFAULT_TRACKER, # Will be throwing away tracks
                                    interval,
                                    vid_out,
                                    device,
                                    i,
                                    region,
                                    **track_params
                                )

                    # Redo tracking using our method and detections
                    stripped = strip_tracks(working_dir)
                    track_over_zoom_interval(
                        stripped,
                        ZOOM_KEEP,
                        region,
                        **reference_params
                    )

                    # Write out updated results
                    for timed_det in stripped:
                        save_timed_detections(timed_det, working_dir / LABEL_SUBFOLDER)

            else:       
                for i, interval in enumerate(dji_interval_generator(video_folder / (vid + VIDEO_EXT), 
                                                            keytimes[vid], INTERVAL_SIZE, OVERLAP)):
                    working_dir = track_interval(
                        model,
                        DEFAULT_TRACKER,
                        interval,
                        vid_out,
                        device,
                        i,
                        None,
                        **track_params
                    )

                    # Redo tracking using our method and detections
                    stripped = strip_tracks(working_dir)
                    track_over_zoom_interval(
                        stripped,
                        len(stripped), # Will cause effective no zoom ops
                        (np.zeros(2), 0), # Dummy 
                        **reference_params
                    )
                    # Write out updated results
                    for timed_det in stripped:
                        save_timed_detections(timed_det, working_dir / LABEL_SUBFOLDER)

if __name__ == "__main__":
    # CLI for simple zoom aware tracking
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Evaluate models on dataset.')
    parser.add_argument('--weights', '-w', type=str, nargs='+', required=True, help='Models to benchmark')
    parser.add_argument('--image_file', '-i', type=str, required=True, help='Path to image file.')
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
    zooming = not args.no_zoom

    run_zoom_tracking(checkpoints, image_file, vids, config, out, device, zooming=zooming, zoom_tours=zoom_tours)
