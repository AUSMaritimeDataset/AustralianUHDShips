from pathlib import Path
import yaml

from utils import get_params, TEST, load_model

def eval_on_dataset(
    checkpoints: list[Path], 
    manifest: Path, 
    out_dir: Path, 
    config: Path | None, 
    device: int | str, 
    split: str = TEST, 
    **kwargs
):
    MANIFEST_ROOT = 'path'
    DATASET_NAME = 'name'

    # prepare benchmark parameters
    eval_params = get_params(config, mode=TEST, **kwargs)
    eval_params.update({ # Enforce benchmark behavior
        "save": True, 
        "save_txt": True, 
        "save_conf": True,
    })
    dataset_name = eval_params.pop(DATASET_NAME, "")
    with open(manifest, 'r', encoding='utf-8') as f:
        manifest_data = yaml.safe_load(f)
    dataset_root = Path(manifest_data[MANIFEST_ROOT]).resolve()
    with open(dataset_root / manifest_data[split], 'r', encoding='utf-8') as f:
        images = [dataset_root / image.rstrip() for image in f.readlines()]
    assert len(images)
    
    for checkpoint in checkpoints:
        # Set up output
        run_name = checkpoint.stem
        if dataset_name:
            run_name += f"_{dataset_name}"

        # Warm up model
        model = load_model(checkpoint)
        model.predict(images[0], device=device)

        # Run benchmark
        model.predict(
            dataset_root / manifest_data[split], 
            device=device, 
            project = out_dir, 
            name = run_name, 
            **eval_params
        )
    

if __name__ == "__main__":
    # CLI to train detectors
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Evaluate models on dataset.')
    parser.add_argument('--weights', '-w', type=str, nargs='+', required=True, help='Models to benchmark')
    parser.add_argument('--manifest', '-m', type=str, required=True, help='Path to data manifest.')
    parser.add_argument('--config', '-c', type=str, default='', help='Path to config.')
    parser.add_argument('--out', '-o', type=str, required=True, help='Path to output folder.')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device to run inference on (e.g., 0, 1, cpu).')
    parser.add_argument('--split', '-s', type=str, default=TEST, help='Override split to use for evaluation')
    parser.set_defaults(resume=False)
    args = parser.parse_args()

    # parse the parsel. I am so funny and hot
    device = args.device
    if device != 'cpu':
        device = int(device)
    checkpoints = [Path(checkpoint).resolve() for checkpoint in args.weights]
    manifest = Path(args.manifest).resolve()
    config = Path(args.config).resolve() if args.config else None
    out = Path(args.out).resolve()
    split = args.split

    eval_on_dataset(checkpoints, manifest, out, config, device, split=split)