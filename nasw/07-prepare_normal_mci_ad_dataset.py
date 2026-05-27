import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "download" / "prepared_normal_mci_ad"
DEFAULT_CLASSES = ("AD", "MCI", "Normal")
SPLITS = ("train", "validation", "test")

LABEL_MAP = {
    "1": "Normal",
    "normal": "Normal",
    "Normal": "Normal",
    "NORMAL": "Normal",
    "정상": "Normal",
    "5": "MCI",
    "mci": "MCI",
    "MCI": "MCI",
    "10": "AD",
    "ad": "AD",
    "AD": "AD",
}

PAPER_TYPES = {"IB-APPS", "IB_APPS", "CERAD-K", "SNSB-II"}
NORMALIZED_PAPER_TYPES = {paper_type.replace("_", "-") for paper_type in PAPER_TYPES}
AUDIO_KIND_KEYS = {
    "raw": ("raw-file",),
    "fine": ("fine-file", "noise-cancelling-file"),
    "init": ("init-file",),
}


def canonical_label(value):
    return LABEL_MAP.get(str(value).strip())


def safe_name(value):
    text = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return safe or "unknown"


def normalize_paper_type(value):
    return str(value).strip().replace("_", "-")


def deterministic_split(subject_id, train_ratio, val_ratio):
    digest = hashlib.sha1(str(subject_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "validation"
    return "test"


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def data_block(json_obj):
    data = json_obj.get("DATA", json_obj)
    if isinstance(data, list):
        return data[0] if data else {}
    return data if isinstance(data, dict) else {}


def detect_paper_type(path, data):
    paper_type = str(data.get("paper_type", "")).strip()
    if paper_type:
        return normalize_paper_type(paper_type)
    for part in path.parts:
        normalized_part = normalize_paper_type(part)
        if normalized_part in NORMALIZED_PAPER_TYPES:
            return normalized_part
    return "unknown"


def file_records(data):
    records = {}
    files = data.get("files", [])
    if isinstance(files, dict):
        files = [files]
    for entry in files:
        if not isinstance(entry, dict):
            continue
        file_type = entry.get("file_type")
        if any(file_type in keys for keys in AUDIO_KIND_KEYS.values()):
            records[file_type] = entry
        for key, value in entry.items():
            if any(key in keys for keys in AUDIO_KIND_KEYS.values()) and isinstance(value, dict):
                records[key] = value
    return records


def audio_path_from_record(json_dir, record):
    file_name = str(record.get("file_name") or "").strip()
    file_path = str(record.get("file_path") or record.get("path") or "").strip()
    file_ext = str(record.get("file_ext", "")).strip().lstrip(".")
    if not file_name and not file_path:
        return None

    name_path = Path(file_name or file_path)
    path_part = Path(file_path) if file_path else None
    candidates = []
    if name_path.is_absolute():
        candidates.append(name_path)
    elif path_part and path_part.is_absolute() and file_name:
        candidates.append(path_part / file_name)
        if file_ext and not Path(file_name).suffix:
            candidates.append(path_part / f"{file_name}.{file_ext}")
    elif path_part and file_name:
        candidates.append(json_dir / path_part / file_name)
        if file_ext and not Path(file_name).suffix:
            candidates.append(json_dir / path_part / f"{file_name}.{file_ext}")
    elif name_path.suffix:
        candidates.append(json_dir / name_path)
    elif file_ext:
        candidates.append(json_dir / f"{file_name}.{file_ext}")
    if not name_path.is_absolute():
        candidates.extend(json_dir.glob(f"{file_name}.*"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def fallback_audio_candidates(json_path):
    stem = json_path.stem
    candidates = []
    for suffix in ("_R", "_F", "_I"):
        if stem.endswith(suffix):
            base = stem[:-2]
            candidates.extend([
                json_path.with_name(f"{base}{suffix}.flac"),
                json_path.with_name(f"{base}{suffix}.wav"),
            ])
    candidates.extend([
        json_path.with_suffix(".flac"),
        json_path.with_suffix(".wav"),
    ])
    return candidates


def choose_audio(json_path, data, audio_kind):
    records = file_records(data)
    if audio_kind == "auto":
        preference = ["raw", "fine", "init"]
    else:
        preference = [audio_kind, "raw", "fine", "init"]

    seen = set()
    for kind in preference:
        if kind in seen:
            continue
        seen.add(kind)
        for record_key in AUDIO_KIND_KEYS[kind]:
            record = records.get(record_key)
            if record:
                path = audio_path_from_record(json_path.parent, record)
                if path and path.exists():
                    return path, kind

    for candidate in fallback_audio_candidates(json_path):
        if candidate.exists():
            suffix = candidate.stem.rsplit("_", 1)[-1]
            kind = {"R": "raw", "F": "fine", "I": "init"}.get(suffix, "unknown")
            return candidate, kind

    return None, None


def infer_source_task(path, data, paper_type):
    if paper_type != "IB-APPS":
        return paper_type
    for value in (data.get("file_seq"), path.stem):
        match = re.search(r"_(\d+)(?:_[RFI])?$", str(value))
        if match:
            return match.group(1)
    return "unknown"


def output_task_name(source_task, paper_type, task_offset):
    if paper_type == "IB-APPS" and str(source_task).isdigit():
        return str(int(source_task) + task_offset)
    return safe_name(source_task)


def save_mel_spectrogram(audio_path, image_path, sr, seconds, dpi):
    y, sr = librosa.load(audio_path, sr=sr)
    max_samples = sr * seconds
    y_segment = y[:max_samples] if len(y) > max_samples else y

    mel = librosa.feature.melspectrogram(y=y_segment, sr=sr)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(1, 1))
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max))
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig(image_path, dpi=dpi)
    plt.close()


def copy_image(source, destination, overwrite):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    shutil.copy2(source, destination)


def validate_split_ratios(train_ratio, val_ratio):
    if not 0 < train_ratio < 1:
        raise SystemExit("--train-ratio must be greater than 0 and less than 1.")
    if not 0 <= val_ratio < 1:
        raise SystemExit("--val-ratio must be greater than or equal to 0 and less than 1.")
    if train_ratio + val_ratio >= 1:
        raise SystemExit("--train-ratio + --val-ratio must be less than 1 so a test split remains.")


def ensure_class_dirs(output_root, tasks):
    for task in sorted(tasks):
        for split in SPLITS:
            for label in DEFAULT_CLASSES:
                (output_root / "multiclass" / task / split / label).mkdir(parents=True, exist_ok=True)
                (output_root / "normal_vs_others" / task / split / ("Normal" if label == "Normal" else "OTHERS")).mkdir(
                    parents=True,
                    exist_ok=True,
                )
                if label in {"MCI", "AD"}:
                    (output_root / "mci_vs_ad" / task / split / label).mkdir(parents=True, exist_ok=True)


def write_preprocessing_config(path, args, counts):
    config = {
        "source_root": str(args.source_root or ""),
        "metadata_csv": str(args.metadata_csv or ""),
        "audio_kind": args.audio_kind,
        "sample_rate": args.sample_rate,
        "seconds": args.seconds,
        "dpi": args.dpi,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "paper_types": args.paper_types,
        "task_offset": args.task_offset,
        "counts": counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def iter_source_jsons(source_root, paper_types):
    source_root = Path(source_root)
    seen = set()
    for paper_type in paper_types:
        normalized_paper_type = normalize_paper_type(paper_type)
        direct_candidates = [source_root / paper_type, source_root / normalized_paper_type]
        if normalized_paper_type == "IB-APPS":
            direct_candidates.append(source_root / "IB_APPS")
        roots = [path for path in direct_candidates if path.exists()]
        if not roots:
            roots = [
                path
                for path in source_root.rglob("*")
                if path.is_dir() and normalize_paper_type(path.name) == normalized_paper_type
            ]
        for root in roots:
            root_key = root.resolve() if root.exists() else root
            if root_key in seen:
                continue
            seen.add(root_key)
            if root.exists():
                yield from sorted(root.rglob("*.json"))


def rows_from_source(args):
    rows = []
    for json_path in iter_source_jsons(args.source_root, args.paper_types):
        try:
            obj = load_json(json_path)
        except Exception as exc:
            rows.append({"error": f"json_error:{type(exc).__name__}", "json_path": str(json_path)})
            continue

        data = data_block(obj)
        label = canonical_label(data.get("subject_type", ""))
        paper_type = detect_paper_type(json_path, data)
        source_task = infer_source_task(json_path, data, paper_type)
        task = output_task_name(source_task, paper_type, args.task_offset)
        audio_path, audio_kind = choose_audio(json_path, data, args.audio_kind)

        rows.append({
            "json_path": str(json_path),
            "audio_path": str(audio_path) if audio_path else "",
            "audio_kind": audio_kind or "",
            "paper_type": paper_type,
            "source_task": source_task,
            "task": task,
            "test_idx": str(data.get("test_idx", "")).strip(),
            "file_seq": str(data.get("file_seq", "")).strip(),
            "subject_type": str(data.get("subject_type", "")).strip(),
            "label": label or "",
            "birth_year": str(data.get("birth_year") or str(data.get("birth_date", ""))[:4]).strip(),
            "gender": str(data.get("gender", "")).strip(),
            "simple_grade": str(data.get("simple_grade", "")).strip(),
            "cdr_score": str(data.get("cdr_score", "")).strip(),
            "error": "",
        })
    return rows


def rows_from_metadata_csv(args):
    rows = []
    with Path(args.metadata_csv).open("r", encoding=args.encoding, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("Metadata CSV has no header row.")

        required = [args.path_column, args.label_column]
        missing = [col for col in required if col and col not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing required CSV columns: {', '.join(missing)}")

        for row in reader:
            raw_audio = row.get(args.path_column, "").strip()
            audio_path = Path(raw_audio)
            if not audio_path.is_absolute():
                audio_path = Path(args.audio_root) / audio_path
            label = canonical_label(row.get(args.label_column, ""))
            paper_type = normalize_paper_type(row.get(args.paper_type_column, "")) if args.paper_type_column else "unknown"
            source_task = row.get(args.task_column, "").strip() if args.task_column else "unknown"
            task = output_task_name(source_task, paper_type, args.task_offset)
            rows.append({
                "json_path": "",
                "audio_path": str(audio_path),
                "audio_kind": "csv",
                "paper_type": paper_type,
                "source_task": source_task,
                "task": task,
                "test_idx": row.get(args.subject_column, "").strip() if args.subject_column else audio_path.stem,
                "file_seq": row.get(args.file_seq_column, "").strip() if args.file_seq_column else audio_path.stem,
                "subject_type": row.get(args.label_column, "").strip(),
                "label": label or "",
                "birth_year": (row.get("birth_year") or row.get("birth_date", "")[:4]).strip(),
                "gender": row.get("gender", "").strip(),
                "simple_grade": row.get("simple_grade", "").strip(),
                "cdr_score": row.get("cdr_score", "").strip(),
                "error": "",
            })
    return rows


def write_manifest(path, rows):
    fieldnames = [
        "json_path",
        "audio_path",
        "audio_kind",
        "paper_type",
        "source_task",
        "task",
        "test_idx",
        "file_seq",
        "subject_type",
        "label",
        "split",
        "birth_year",
        "gender",
        "simple_grade",
        "cdr_score",
        "mel_image_path",
        "multiclass_path",
        "multiclass_all_path",
        "normal_vs_others_path",
        "mci_vs_ad_path",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(args):
    if not args.source_root and not args.metadata_csv:
        raise SystemExit("Provide either --source-root or --metadata-csv.")
    validate_split_ratios(args.train_ratio, args.val_ratio)

    print(f"source_root: {args.source_root or '-'}")
    print(f"output_root: {args.output_root}")
    print(f"audio_kind: {args.audio_kind}")

    output_root = Path(args.output_root)
    raw_rows = rows_from_source(args) if args.source_root else rows_from_metadata_csv(args)
    completed = []
    written_tasks = {"ALL"}
    counts = {
        "read": len(raw_rows),
        "written": 0,
        "skipped_unknown_label": 0,
        "skipped_missing_audio": 0,
        "skipped_task_filter": 0,
        "json_errors": 0,
    }

    for row in raw_rows:
        if row.get("error"):
            counts["json_errors"] += 1
            row["split"] = ""
            completed.append(row)
            continue

        label = row["label"]
        if not label:
            counts["skipped_unknown_label"] += 1
            row["error"] = "unknown_label"
            row["split"] = ""
            completed.append(row)
            continue

        task = safe_name(row["task"])
        if args.tasks and task not in args.tasks:
            counts["skipped_task_filter"] += 1
            continue
        written_tasks.add(task)

        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            counts["skipped_missing_audio"] += 1
            row["error"] = "missing_audio"
            row["split"] = ""
            completed.append(row)
            continue

        subject_id = row["test_idx"] or audio_path.stem
        split = deterministic_split(subject_id, args.train_ratio, args.val_ratio)
        image_name = f"{safe_name(subject_id)}__{task}__{safe_name(audio_path.stem)}.jpg"

        mel_path = output_root / "mel_images" / task / label / image_name
        if not args.manifest_only and (args.overwrite or not mel_path.exists()):
            save_mel_spectrogram(audio_path, mel_path, args.sample_rate, args.seconds, args.dpi)

        multiclass_path = output_root / "multiclass" / task / split / label / image_name
        multiclass_all_path = output_root / "multiclass" / "ALL" / split / label / image_name
        normal_vs_others_label = "Normal" if label == "Normal" else "OTHERS"
        normal_vs_others_path = (
            output_root / "normal_vs_others" / task / split / normal_vs_others_label / image_name
        )
        mci_vs_ad_path = ""
        if label in {"MCI", "AD"}:
            mci_vs_ad_path = str(output_root / "mci_vs_ad" / task / split / label / image_name)

        if not args.manifest_only:
            copy_image(mel_path, multiclass_path, args.overwrite)
            copy_image(mel_path, multiclass_all_path, args.overwrite)
            copy_image(mel_path, normal_vs_others_path, args.overwrite)
            if mci_vs_ad_path:
                copy_image(mel_path, Path(mci_vs_ad_path), args.overwrite)

        row.update({
            "split": split,
            "mel_image_path": str(mel_path),
            "multiclass_path": str(multiclass_path),
            "multiclass_all_path": str(multiclass_all_path),
            "normal_vs_others_path": str(normal_vs_others_path),
            "mci_vs_ad_path": mci_vs_ad_path,
            "error": "",
        })
        completed.append(row)
        counts["written"] += 1

    manifest_path = output_root / "dataset_manifest.csv"
    if not args.manifest_only:
        ensure_class_dirs(output_root, written_tasks)
    write_preprocessing_config(output_root / "preprocessing_config.json", args, counts)
    write_manifest(manifest_path, completed)
    print(f"manifest: {manifest_path}")
    for key, value in counts.items():
        print(f"{key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build Normal/MCI/AD mel-spectrogram datasets from the NIA cognitive "
            "voice source tree or from a metadata CSV."
        )
    )
    parser.add_argument(
        "--source-root",
        default=os.environ.get("VELORA_DATASET_ROOT", str(DEFAULT_SOURCE_ROOT)),
        help="NIA source root containing IB-APPS/CERAD-K/SNSB-II. Defaults to ../dataset from this script.",
    )
    parser.add_argument("--metadata-csv", default=None, help="Optional CSV alternative to --source-root.")
    parser.add_argument("--audio-root", default=".", help="Base directory for relative CSV audio paths.")
    parser.add_argument(
        "--output-root",
        default=os.environ.get("VELORA_PREPARED_ROOT", str(DEFAULT_OUTPUT_ROOT)),
        help="Output dataset directory. Defaults to ./download/prepared_normal_mci_ad from this script.",
    )
    parser.add_argument(
        "--paper-types",
        nargs="+",
        default=["IB-APPS", "CERAD-K", "SNSB-II"],
        help="Source paper types to scan.",
    )
    parser.add_argument(
        "--audio-kind",
        choices=["raw", "fine", "init", "auto"],
        default="raw",
        help="Which audio variant to use. Guide labels raw-file as the learning source file.",
    )
    parser.add_argument(
        "--task-offset",
        type=int,
        default=1,
        help="Add this offset to IB-APPS source task numbers. Use 1 to convert 0-10 into 1-11.",
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional output task whitelist.")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=100, help="1x1 inch image at 100 dpi creates 100x100 images.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--manifest-only", action="store_true", help="Only write manifest; do not create images.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--path-column", default="file_path")
    parser.add_argument("--label-column", default="subject_type")
    parser.add_argument("--subject-column", default="test_idx")
    parser.add_argument("--task-column", default="task")
    parser.add_argument("--paper-type-column", default="paper_type")
    parser.add_argument("--file-seq-column", default="file_seq")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
