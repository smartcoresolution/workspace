import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "dataset"

LABELS = {
    "Normal": "1",
    "MCI": "5",
    "AD": "10",
}


def safe_remove(path):
    if path.exists():
        shutil.rmtree(path)


def synth_voice_like_audio(label, index, sample_rate, seconds):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    base_freq = {"Normal": 180.0, "MCI": 150.0, "AD": 120.0}[label]
    modulation = {"Normal": 4.0, "MCI": 2.2, "AD": 1.2}[label]
    rng = np.random.default_rng(index + int(base_freq))

    carrier = np.sin(2 * math.pi * base_freq * t)
    harmonic = 0.35 * np.sin(2 * math.pi * base_freq * 2.0 * t)
    envelope = 0.5 + 0.5 * np.sin(2 * math.pi * modulation * t)
    pauses = (np.sin(2 * math.pi * (0.35 + index * 0.03) * t) > -0.35).astype(np.float32)
    noise = 0.015 * rng.normal(size=t.shape)

    audio = (carrier + harmonic) * envelope * pauses + noise
    audio = audio / max(np.max(np.abs(audio)), 1e-6) * 0.35
    return audio.astype(np.float32)


def write_audio_triplet(subject_dir, stem, label, index, sample_rate, seconds):
    audio = synth_voice_like_audio(label, index, sample_rate, seconds)
    for suffix in ("I", "R", "F"):
        sf.write(subject_dir / f"{stem}_{suffix}.flac", audio, sample_rate)


def write_json(subject_dir, stem, label, paper_type, test_idx, file_seq):
    payload = {
        "DATA": {
            "file_seq": file_seq,
            "test_idx": test_idx,
            "paper_type": paper_type,
            "birth_year": "1975",
            "gender": "1",
            "simple_grade": "27",
            "subject_type": LABELS[label],
            "cdr_score": "",
            "files": [
                {
                    "init-file": {
                        "file_name": f"{stem}_I",
                        "file_ext": "flac",
                    }
                },
                {
                    "raw-file": {
                        "file_name": f"{stem}_R",
                        "file_ext": "flac",
                    }
                },
                {
                    "fine-file": {
                        "file_name": f"{stem}_F",
                        "file_ext": "flac",
                    }
                },
            ],
            "script": [
                {
                    "alternatives": [
                        {
                            "transcript": "샘플 음성 데이터입니다.",
                            "confidence": "1.0",
                            "words": [
                                {"startTime": "0.0", "endTime": "1.0", "word": "샘플"},
                                {"startTime": "1.0", "endTime": "2.0", "word": "음성"},
                            ],
                        }
                    ]
                }
            ],
        }
    }
    with (subject_dir / f"{stem}_R.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def create_ib_apps(output_root, label, label_index, subjects_per_label, sample_rate, seconds):
    root = output_root / "IB-APPS"
    for subject_no in range(subjects_per_label):
        test_idx = f"sample-{label.lower()}-{subject_no:03d}"
        subject_dir = root / test_idx
        subject_dir.mkdir(parents=True, exist_ok=True)
        for task in range(2):
            stem = f"{test_idx}_{task}"
            file_seq = stem
            audio_index = label_index * 1000 + subject_no * 10 + task
            write_audio_triplet(subject_dir, stem, label, audio_index, sample_rate, seconds)
            write_json(subject_dir, stem, label, "IB_APPS", test_idx, file_seq)


def create_long_exam(output_root, paper_type, label, label_index, subjects_per_label, sample_rate, seconds):
    root = output_root / paper_type
    for subject_no in range(subjects_per_label):
        test_idx = f"sample-{paper_type.lower()}-{label.lower()}-{subject_no:03d}"
        subject_dir = root / test_idx
        subject_dir.mkdir(parents=True, exist_ok=True)
        stem = test_idx
        audio_index = label_index * 2000 + subject_no
        write_audio_triplet(subject_dir, stem, label, audio_index, sample_rate, seconds)
        write_json(subject_dir, stem, label, paper_type, test_idx, stem)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a tiny fake NIA-style dataset for pipeline smoke tests."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--subjects-per-label", type=int, default=30)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    if args.overwrite:
        for name in ("IB-APPS", "CERAD-K", "SNSB-II"):
            safe_remove(output_root / name)

    output_root.mkdir(parents=True, exist_ok=True)
    for label_index, label in enumerate(LABELS):
        create_ib_apps(output_root, label, label_index, args.subjects_per_label, args.sample_rate, args.seconds)
        create_long_exam(output_root, "CERAD-K", label, label_index, args.subjects_per_label, args.sample_rate, args.seconds)
        create_long_exam(output_root, "SNSB-II", label, label_index, args.subjects_per_label, args.sample_rate, args.seconds)

    print(f"sample_dataset: {output_root}")
    print("created: IB-APPS, CERAD-K, SNSB-II")
    print("labels: Normal=1, MCI=5, AD=10")


if __name__ == "__main__":
    main()
