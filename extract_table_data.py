from pathlib import Path
import  json







DET_OUT = 'DetectorMetrics'
DET_TABLE = 'det_tab.txt'



DET_METRICS = "f1@50", "ap@50"
MEAN = 'mean'
CLS = 'class'
def det_tables(det_out: Path, tab: Path):
    lines = []
    for folder in det_out.glob('*.json'):
        run_name = folder.stem
        with open(folder, 'r', encoding='utf-8') as f:
            run_data = json.load(f)[MEAN][CLS][MEAN]
        line = f"{run_name} > "
        for metric in DET_METRICS:
            line += f"{metric}: {run_data[metric]}"
        lines.append(line)
    with open(tab, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

TRACK_OIT = 'TrackerMetrics'
TRACK_TAB = 'track_tab.txt'
DESIRED = "without_blank"
DESIRED_PROPER = "tracked_of_detected"
def track_tables(track_out: Path, tab: Path):
    lines = []
    for file in track_out.glob('*.json'):
        with open(file, 'r', encoding='utf-8') as f:
            val = json.load(f)[MEAN][DESIRED][DESIRED_PROPER]
        lines.append(f'{file.stem}: {val}')
    with open(tab, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def extract_tables(out_root: Path, out_out: Path):
    out_out.mkdir(exist_ok=True)
    det_tables(out_root / DET_OUT, out_out / DET_TABLE)
    track_tables(out_root / TRACK_OIT, out_out / TRACK_TAB)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Extract Tables')
    parser.add_argument('--root', '-i', type=str, required=True, help='Out root')
    parser.add_argument('--out', '-o', type=str, required=True, help='Output folder')
    args = parser.parse_args()

    # HIGHRES = np.array([3840, 2160])
    extract_tables(Path(args.root).resolve(), Path(args.out).resolve())