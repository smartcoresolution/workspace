import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "dataset"
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "download" / "prepared_normal_mci_ad"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "download" / "ad_mci_normal"


def run(command):
    print("\n$", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the raw NIA dataset and train the AD/MCI/Normal VGG16 model."
    )
    parser.add_argument("--source-root", default=os.environ.get("VELORA_DATASET_ROOT", str(DEFAULT_SOURCE_ROOT)))
    parser.add_argument("--prepared-root", default=os.environ.get("VELORA_PREPARED_ROOT", str(DEFAULT_PREPARED_ROOT)))
    parser.add_argument("--output-root", default=os.environ.get("VELORA_MODEL_OUTPUT_ROOT", str(DEFAULT_MODEL_ROOT)))
    parser.add_argument("--tasks", nargs="*", default=["ALL"])
    parser.add_argument("--audio-kind", choices=["raw", "fine", "init", "auto"], default="raw")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing final/best model output.")
    parser.add_argument("--resume-from", choices=["final", "best"], default="final")
    parser.add_argument("--initial-model", default=None, help="Optional .h5 model to start training from.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_prepare:
        prepare_command = [
            sys.executable,
            str(PROJECT_ROOT / "07-prepare_normal_mci_ad_dataset.py"),
            "--source-root",
            args.source_root,
            "--output-root",
            args.prepared_root,
            "--audio-kind",
            args.audio_kind,
            "--sample-rate",
            str(args.sample_rate),
            "--seconds",
            str(args.seconds),
        ]
        if args.overwrite:
            prepare_command.append("--overwrite")
        run(prepare_command)

    train_command = [
        sys.executable,
        str(PROJECT_ROOT / "08-train_normal_mci_ad_vgg16.py"),
        "--dataset-root",
        str(Path(args.prepared_root) / "multiclass"),
        "--output-root",
        args.output_root,
        "--tasks",
        *args.tasks,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--weights",
        args.weights,
    ]
    if args.resume:
        train_command.append("--resume")
        train_command.extend(["--resume-from", args.resume_from])
    if args.initial_model:
        train_command.extend(["--initial-model", args.initial_model])
    run(train_command)


if __name__ == "__main__":
    main()
