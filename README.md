# Australian UHD Ships

Gday! This is the official repository for Australian UHD Ships. It contains the implementation of our paper: [AUS: An Egocentric Maritime Zoom Aware Tracking Benchmark](https://openreview.net/forum?id=EgGQVd5M8a). The repository also holds the [supplementary material for our paper](TrackingImplmentation.pdf)

## Download our Dataset
Australian UHD Ships is too large (>100GB) to be shared in this repo. It will be shared upon request via the review system while our paper is under review. Once our paper is accepted, we will provide an institutional cloud drive link here.

## Supplementary Material
A description of our zoom aware tracking algorithm is given [here](TrackingImplmentation.pdf)

## Run Benchmark
This section describes how to replicate the results of our paper. The pipeline should be easy to extend to your own datasets and models. All code snippets assume they are being run from the root of this repository. Our environment is tested on an x86 Ubuntu 24.04 Desktop.

### Set up environment
Install conda, then prepare environment:
```bash
conda env create -p ./.conda -f bench_env.yml
conda activate ./.conda
```

### Prepare datasets
Download [ABOShips-PLUS](https://zenodo.org/records/10469672) and convert it:
```bash
python ./conversion_scripts/ABO_to_YOLO.py -i <<PATH_TO_DOWNLOADED_DATASET>> -o ./datasets/ABOShips_PLUS
```

Download [SeaClips](https://huggingface.co/datasets/SEA-AI/SeaClips) and convert it:
```bash
python ./conversion_scripts/Seaclips_to_YOLO.py -i <<PATH_TO_DOWNLOADED_DATASET>> -o ./datasets/SeaClips
```

Download [SingaporeMaritime](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset) and convert it:
```bash
python ./conversion_scripts/SingaporeMaritime_to_YOLO.py -i <<PATH_TO_DOWNLOADED_DATASET>> -o ./datasets/SingaporeMaritime
```

Download [WSODD](https://pan.baidu.com/s/1-xT6fwH3alW78uCsm9VjRA) and convert it:
```bash
python ./conversion_scripts/WSODD_to_YOLO.py -i <<PATH_TO_DOWNLOADED_DATASET>> -o ./datasets/WSODD
```

Download [AUS UHD Ships](#download-our-dataset) and save it as `./datasets/AustralianUHDShips`

### Train detectors
Download [YOLOv26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) and [RT-DETER-l](https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt) and save them as `./models/base_weights/yolo26n.pt` and `./models/base_weights/yolo/rtdetr-l.pt` respectively.

`train_detectors.py` provides an interface to train detectors on converted datasets. We provide a helpful shell script to train all required detectors and prepare the appropriate file structure for the benchmark. Run:
```bash
chmod +x ./train_detectors.sh
./train_detectors.sh
```

### Run Dataset Generalisation Benchmark
`eval_on_dataset.py` provides an interface for running a trained detector on a dataset. `metrics.py` provides an interface for evaluating the performance of a given run. We provide a helpful shell script to replicate our benchmark results, assuming  the file structure described here has been followed. Run:
```bash
chmod +x ./generalisation_benchmark.sh
./generalisation_benchmark.sh
```

### Run Zoom Aware Tracking Benchmark
`eval_tracker.py` provides an interface for running a tracker over zoom intervals. `zoom_aware_tracking.py` provides an interface to run our  zoom aware tracking reference method. `metrics.py` provides an interface for evaluating the performance of a given run. We provide a helpful shell script to replicate our benchmark results, assuming  the file structure described here has been followed. Run:
```bash
chmod +x ./tracking_benchmark.sh
./tracking_benchmark.sh
```