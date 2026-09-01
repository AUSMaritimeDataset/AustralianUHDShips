from pathlib import Path
import json

from utils import load_model, get_params, TRAIN

SEED = 42069

# TODO docstrings throughout files
def train_models(
        checkpoints: list[Path], 
        manifest: Path, 
        out: Path,
        device: int | str = 'cpu',
        config: Path | None = None,
        **kwargs
):
    training_params = get_params(config, mode=TRAIN, **kwargs)
    dataset_name = training_params.pop("name", "")
    
    # TODO docstring
    for checkpoint in checkpoints:
        run_name = checkpoint.stem
        if dataset_name:
            run_name += f"_{dataset_name}"

        model = load_model(checkpoint)
        results = model.train(
            data=str(manifest),
            project=out,
            name=run_name,
            device=device,
            seed=SEED,
            **training_params
        )
        del results
        del model


if __name__ == "__main__":
    # CLI to train detectors
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Train on dataset.')
    parser.add_argument('--weights', '-w', type=str, nargs='+', required=True, help='Checkpoint files to finetune from')
    parser.add_argument('--manifest', '-m', type=str, required=True, help='Path to data manifest.')
    parser.add_argument('--config', '-c', type=str, default='', help='Path to training config.')
    parser.add_argument('--out', '-o', type=str, required=True, help='Path to output folder.')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device to run inference on (e.g., 0, 1, cpu).')
    parser.add_argument('--resume', '-r', action='store_true', help='weights are a checkpoint of existing run')
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
    resume = args.resume

    if len(checkpoints) > 1: # TODO fix
        raise NotImplementedError("Multiple checkpoints causes a cache error after first model. Run training for each model as its own call")
    train_models(checkpoints, manifest, out, device, config, resume=resume)