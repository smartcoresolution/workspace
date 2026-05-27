import argparse
import io
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "download" / "ad_mci_normal"
DEFAULT_CLASSES = ["AD", "MCI", "Normal"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer AD/MCI/Normal probabilities from one wav/flac file and a trained 3-class model."
    )
    parser.add_argument("--audio-file", required=True, help="Input wav/flac file.")
    parser.add_argument(
        "--model",
        default=os.environ.get("VELORA_COGNITIVE_MODEL_PATH"),
        help="Trained .h5 model path. Defaults to VELORA_COGNITIVE_MODEL_PATH or latest best model under ./download/ad_mci_normal.",
    )
    parser.add_argument(
        "--metadata-json",
        default=os.environ.get("VELORA_COGNITIVE_METADATA_PATH"),
        help="Optional metadata JSON from 08-train_normal_mci_ad_vgg16.py.",
    )
    parser.add_argument("--class-names", nargs="+", default=None, help="Fallback class order when metadata is absent.")
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--seconds", type=int, default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--json-output", default=None, help="Optional path to save prediction JSON.")
    return parser.parse_args()


def find_latest_model(model_root):
    root = Path(model_root)
    patterns = [
        "task-ALL/*_best.h5",
        "task-ALL/*.h5",
        "task-*/*_best.h5",
        "task-*/*.h5",
    ]
    for pattern in patterns:
        matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def infer_metadata_path(model_path):
    model = Path(model_path)
    candidates = []
    if model.name.endswith("_best.h5"):
        candidates.append(model.with_name(model.name.replace("_best.h5", "_metadata.json")))
    if model.name.endswith("_final.h5"):
        candidates.append(model.with_name(model.name.replace("_final.h5", "_metadata.json")))
    candidates.append(model.with_suffix(".json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_metadata(metadata_json):
    if not metadata_json:
        return {}
    with Path(metadata_json).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_class_names(metadata, fallback_names, output_dim):
    class_indices = metadata.get("class_indices")
    if class_indices:
        names = [None] * len(class_indices)
        for name, index in class_indices.items():
            names[int(index)] = name
        if all(names):
            return names
    class_names = metadata.get("class_names")
    if class_names:
        return class_names

    if fallback_names:
        return fallback_names
    if output_dim == 3:
        return DEFAULT_CLASSES
    return [f"class_{index}" for index in range(output_dim)]


def audio_to_array(audio_file, sample_rate, seconds, image_size, dpi):
    y, sr = librosa.load(audio_file, sr=sample_rate)
    max_samples = sr * seconds
    y_segment = y[:max_samples] if len(y) > max_samples else y

    mel = librosa.feature.melspectrogram(y=y_segment, sr=sr)
    fig = plt.figure(figsize=(image_size[1] / dpi, image_size[0] / dpi), dpi=dpi)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max))
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    buffer.seek(0)

    image = Image.open(buffer).convert("RGB").resize((image_size[1], image_size[0]))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def save_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    model_path = Path(args.model) if args.model else find_latest_model(DEFAULT_MODEL_ROOT)
    if not model_path:
        raise SystemExit(
            "No model was provided and no .h5 model was found under "
            f"{DEFAULT_MODEL_ROOT}. Train first or pass --model."
        )

    metadata_path = Path(args.metadata_json) if args.metadata_json else infer_metadata_path(model_path)
    metadata = load_metadata(metadata_path)
    model = tf.keras.models.load_model(model_path)
    preprocessing = metadata.get("preprocessing") or {}
    image_size = tuple(args.image_size or metadata.get("image_size") or [100, 100])
    sample_rate = args.sample_rate if args.sample_rate is not None else int(preprocessing.get("sample_rate", 48000))
    seconds = args.seconds if args.seconds is not None else int(preprocessing.get("seconds", 30))
    dpi = args.dpi if args.dpi is not None else int(preprocessing.get("dpi", 100))
    array = audio_to_array(args.audio_file, sample_rate, seconds, image_size, dpi)

    probabilities = np.asarray(model.predict(array, verbose=0)[0]).reshape(-1)
    if len(probabilities) != 3:
        raise SystemExit(
            "This script expects a 3-class softmax model for AD/MCI/Normal. "
            f"The loaded model returned {len(probabilities)} value(s)."
        )

    class_names = load_class_names(metadata, args.class_names, len(probabilities))
    if len(class_names) != len(probabilities):
        raise SystemExit(
            f"Class count mismatch: model outputs {len(probabilities)} values, "
            f"but class names are {class_names}."
        )

    prediction = {
        name: float(probabilities[index])
        for index, name in enumerate(class_names)
    }
    predicted_label = max(prediction, key=prediction.get)
    payload = {
        "audio_file": str(Path(args.audio_file)),
        "model": str(model_path),
        "metadata_json": str(metadata_path) if metadata_path else "",
        "sample_rate": sample_rate,
        "seconds": seconds,
        "dpi": dpi,
        "image_size": list(image_size),
        "predicted_label": predicted_label,
        "probabilities": prediction,
    }

    print(f"model: {model_path}")
    print(f"predicted_label: {predicted_label}")
    for name, value in sorted(prediction.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {value:.6f}")

    if args.json_output:
        save_json(args.json_output, payload)
        print(f"json_output: {args.json_output}")


if __name__ == "__main__":
    main()
