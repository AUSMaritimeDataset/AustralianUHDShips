#!/bin/bash
DATASET_NAME="AustralianUHDShips"
MODEL_NAME="yolo26n"

BYTENAME="bytetrack"
FASTNAME="fasttrack"
OCSTNAME="ocsort"
LOC_TRACKERS=( $BYTENAME $FASTNAME $OCSTNAME)

BOTSNAME="botsort"
DPOCNAME="deepocsort"
TKTKNAME="tracktrack"
APP_TRACKERS=( $BOTSNAME $DPOCNAME $TKTKNAME)

TOUR_OUT="./out/StandardTours.json" # Can be wherever you want it
INFERENCE_OUT="./out/TrackerRuns"  # As above
METRICS_OUT="./out/TrackerMetrics" # As above
mkdir "${INFERENCE_OUT}"
mkdir "${METRICS_OUT}"

# Compute Standard Tours for benchmark
python zoom_sim.py -i "./datasets/${DATASET_NAME}/keyframes/all.txt" -c "./configs/${DATASET_NAME}.json" -o "${TOUR_OUT}" -n 3

# Localisation only trackers 
# Inference
for tracker in "${LOC_TRACKERS[@]}"
do
    # Baseline
    python eval_tracker.py -w  "./models/trained_weights/${MODEL_NAME}_${DATASET_NAME}.pt" -t "./models/trackers/${tracker}.yaml" -i "./datasets/${DATASET_NAME}/keyframes/all.txt" -v "./datasets/${DATASET_NAME}/videos" -c "./configs/${DATASET_NAME}.json" -o "${INFERENCE_OUT}/LocTrackers" -d 0 --no_zoom
    # With Zoom
    python eval_tracker.py -w  "./models/trained_weights/${MODEL_NAME}_${DATASET_NAME}.pt" -t "./models/trackers/${tracker}.yaml" -i "./datasets/${DATASET_NAME}/keyframes/all.txt" -v "./datasets/${DATASET_NAME}/videos" -c "./configs/${DATASET_NAME}.json" -o "${INFERENCE_OUT}/LocTrackersZoom" -d 0 -z "${TOUR_OUT}"
done
# Metrics
python metrics.py -d "${INFERENCE_OUT}/LocTrackers/${MODEL_NAME}_${DATASET_NAME}_${DATASET_NAME}" -m "./datasets/${DATASET_NAME}/keyframes/manifest.yaml" -s bench -c "./configs/${DATASET_NAME}.json" -o "${METRICS_OUT}" --tracking
for tracker in "${LOC_TRACKERS[@]}"
do
    mv "${METRICS_OUT}/${tracker}.json" "${METRICS_OUT}/${tracker}_baseline.json"
done
python metrics.py -d "${INFERENCE_OUT}/LocTrackersZoom/${MODEL_NAME}_${DATASET_NAME}_${DATASET_NAME}" -m "./datasets/${DATASET_NAME}/keyframes/manifest.yaml" -s bench -c "./configs/${DATASET_NAME}.json" -o "${METRICS_OUT}" --tracking
for tracker in "${LOC_TRACKERS[@]}"
do
    mv "${METRICS_OUT}/${tracker}.json" "${METRICS_OUT}/${tracker}_zoom.json"
done

# Appearance based trackers 
# Inference
for tracker in "${APP_TRACKERS[@]}"
do
    # Baseline
    python eval_tracker.py -w  "./models/trained_weights/${MODEL_NAME}_${DATASET_NAME}.pt" -t "./models/trackers/${tracker}.yaml" -i "./datasets/${DATASET_NAME}/keyframes/all.txt" -v "./datasets/${DATASET_NAME}/videos" -c "./configs/${DATASET_NAME}.json" -o "${INFERENCE_OUT}/AppTrackers" -d 0 --no_zoom
    # With Zoom
    python eval_tracker.py -w  "./models/trained_weights/${MODEL_NAME}_${DATASET_NAME}.pt" -t "./models/trackers/${tracker}.yaml" -i "./datasets/${DATASET_NAME}/keyframes/all.txt" -v "./datasets/${DATASET_NAME}/videos" -c "./configs/${DATASET_NAME}.json" -o "${INFERENCE_OUT}/AppTrackersZoom" -d 0 -z "${TOUR_OUT}"
done
# Metrics
python metrics.py -d "${INFERENCE_OUT}/AppTrackers/${MODEL_NAME}_${DATASET_NAME}_${DATASET_NAME}" -m "./datasets/${DATASET_NAME}/keyframes/manifest.yaml" -s bench -c "./configs/${DATASET_NAME}.json" -o "${METRICS_OUT}" --tracking
for tracker in "${LOC_TRACKERS[@]}"
do
    mv "${METRICS_OUT}/${tracker}.json" "${METRICS_OUT}/${tracker}_baseline.json"
done
python metrics.py -d "${INFERENCE_OUT}/AppTrackersZoom/${MODEL_NAME}_${DATASET_NAME}_${DATASET_NAME}" -m "./datasets/${DATASET_NAME}/keyframes/manifest.yaml" -s bench -c "./configs/${DATASET_NAME}.json" -o "${METRICS_OUT}" --tracking
for tracker in "${LOC_TRACKERS[@]}"
do
    mv "${METRICS_OUT}/${tracker}.json" "${METRICS_OUT}/${tracker}_zoom.json"
done

# Reference Method
# Baseline
python zoom_aware_tracking.py -w  "./models/trained_weights/${MODEL_NAME}_${DATASET_NAME}.pt" -i "./datasets/${DATASET_NAME}/keyframes/all.txt" -v "./datasets/${DATASET_NAME}/videos" -c "./configs/${DATASET_NAME}.json" -o "${INFERENCE_OUT}/ReferenceMethod" -d 0 --no_zoom
python metrics.py -d "${INFERENCE_OUT}/ReferenceMethod" -m "./datasets/${DATASET_NAME}/keyframes/manifest.yaml" -s bench -c "./configs/${DATASET_NAME}.json" -o "${METRICS_OUT}" --tracking
mv "${METRICS_OUT}/${MODEL_NAME}_${DATASET_NAME}_${DATASET_NAME}.json" "${METRICS_OUT}/ReferenceMethod.json"
# With Zoom 
python zoom_aware_tracking.py -w  "./models/trained_weights/${MODEL_NAME}_${DATASET_NAME}.pt" -i "./datasets/${DATASET_NAME}/keyframes/all.txt" -v "./datasets/${DATASET_NAME}/videos" -c "./configs/${DATASET_NAME}.json" -o "${INFERENCE_OUT}/ReferenceMethodZoom" -d 0 -z "${TOUR_OUT}"
python metrics.py -d "${INFERENCE_OUT}/ReferenceMethodZoom" -m "./datasets/${DATASET_NAME}/keyframes/manifest.yaml" -s bench -c "./configs/${DATASET_NAME}.json" -o "${METRICS_OUT}" --tracking
mv "${METRICS_OUT}/${MODEL_NAME}_${DATASET_NAME}_${DATASET_NAME}.json" "${METRICS_OUT}/ReferenceMethodZoom.json"