#!/bin/bash
AUSNAME="AustralianUHDShips"

ABONAME="ABOShips_PLUS"
SEANAME="SeaClips"
SIGNAME="SingaporeMaritime"
WSONAME="WSODD"
DATASETS=( $ABONAME $SEANAME $SIGNAME $WSONAME)

YOLONAME="yolo26n"
RTDTNAME="rtdetr-l"
MODELS=( $YOLONAME $RTDTNAME)

INFERENCE_OUT="./out/DetectorRuns" # Can be wherever you want it
METRICS_OUT="./out/GeneralisationMetrics" # As above
mkdir "${INFERENCE_OUT}"
mkdir "${METRICS_OUT}"



# Run inference on datasets
for model in "${MODELS[@]}"
do
    for dataset in "${DATASETS[@]}"
    do
        # Baseline Inference
        python eval_on_dataset.py -w "./models/trained_weights/${model}_${dataset}.pt" -m "./datasets/${dataset}/manifest.yaml" -c "./configs/${dataset}.json" -o "${INFERENCE_OUT}" -d 0
        mv "${INFERENCE_OUT}/${model}_${dataset}_${dataset}" "${INFERENCE_OUT}/${model}_${dataset}_Baseline"

        # Baseline Metrics
        python metrics.py -d "${INFERENCE_OUT}" -m "./datasets/${dataset}/manifest.yaml" -c "./configs/${dataset}.json" -o "${METRICS_OUT}" -p "'*_${dataset}_Baseline'"

        # Generalisation Inference
        python eval_on_dataset.py -w "./models/trained_weights/${model}_${dataset}.pt" -m "./datasets/${AUSNAME}/keyframes/manifest.yaml" -c "./configs/${dataset}.json" -o "${INFERENCE_OUT}" -d 0 -s bench
        mv "${INFERENCE_OUT}/${model}_${dataset}_${dataset}" "${INFERENCE_OUT}/${model}_${dataset}_${AUSNAME}"

        # Generalisation Metrics
        python metrics.py -d "${INFERENCE_OUT}" -m "./datasets/${AUSNAME}/keyframes/manifest.yaml" -c "./configs/${AUSNAME}.json" -o "${METRICS_OUT}" -p "'*_${dataset}_${AUSNAME}'" -s bench --translate
    done
done