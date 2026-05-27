# VELORA Train

Server-side training, verification, dataset preparation, and model inference utilities for the VELORA cognitive voice pipeline.

## Contents

- `07-prepare_normal_mci_ad_dataset.py`: build `Normal/MCI/AD` training datasets from the NIA source tree.
- `08-train_normal_mci_ad_vgg16.py`: train the current VGG16-based 3-class model.
- `09-infer_normal_mci_ad.py`: run local inference with a trained 3-class `.h5` model.
- `10-train_from_raw_dataset.py`: one-command raw dataset preparation and model training for the server layout.
- `11-create_sample_dataset.py`: create a tiny fake raw dataset for smoke tests before AI Hub data arrives.

## Environment

```bash
conda env create -f environment.yml
conda activate velora-cognitive-voice
```

For pip-only servers:

```bash
pip install -r requirements-server.txt
```

## Packed Conda Environment

To package the prepared conda environment for transfer to another server:

```bash
cd ~/workspace/nasw
mkdir -p download
conda run -n velora-cognitive-voice conda-pack \
  -n velora-cognitive-voice \
  -o download/velora-cognitive-voice.tar.gz \
  --force
```

Use `conda-pack`, not `conda pack`. `conda pack` is not a built-in conda
subcommand on this setup.

To use the packed environment later on another server:

```bash
mkdir -p ~/envs/velora-cognitive-voice
tar -xzf velora-cognitive-voice.tar.gz -C ~/envs/velora-cognitive-voice
source ~/envs/velora-cognitive-voice/bin/activate
conda-unpack
```

Run `conda-unpack` only once after extracting the archive. After that, activate
the environment with:

```bash
source ~/envs/velora-cognitive-voice/bin/activate
```

## Workspace Sync

Keep this workspace layout:

```text
~/workspace/
  dataset/
  nasw/
    download/
```

When syncing `~/workspace` with `smartcoresolution/workspace.git`, exclude the
raw dataset and generated download artifacts. Add this `.gitignore` at
`~/workspace/.gitignore`:

```gitignore
dataset/
nasw/download/
*.tar.gz
*.h5
*.pyc
__pycache__/
```

Then sync only the tracked workspace files:

```bash
cd ~/workspace
git status
git add .gitignore
git add nasw
git commit -m "Sync workspace structure"
git push origin main
```

## Dataset Preparation

On the target server, this repository is expected to live in `~/workspace/nasw`,
with raw source data in `~/workspace/dataset` and generated outputs in
`~/workspace/nasw/download`.

Before the real AI Hub dataset is installed, create a small synthetic dataset
to verify the environment and pipeline:

```bash
cd ~/workspace/nasw
python 11-create_sample_dataset.py --overwrite
python 10-train_from_raw_dataset.py --overwrite --tasks ALL --weights none --epochs 1 --batch-size 4
```

This sample data is only for smoke testing. It must not be used to judge model
accuracy. The sample generator creates enough subjects per class for stable
train/validation/test smoke tests by default.

## Closed-network / Offline validation

Before the real dataset arrives and before the isolated GPU server is closed,
validate the full pipeline locally using the sample dataset and the same
environment.

1. Create or update the local Python environment:

```bash
conda env create -f environment.yml
conda activate velora-cognitive-voice
```

or using pip:

```bash
python -m pip install -r requirements-server.txt
```

2. Create and package the conda environment for offline transfer:

```bash
cd ~/workspace/nasw
conda install -c conda-forge conda-pack
mkdir -p download
conda run -n velora-cognitive-voice conda-pack \
  -n velora-cognitive-voice \
  -o download/velora-cognitive-voice.tar.gz \
  --force
```

This creates a packaged environment tarball under `~/workspace/nasw/download/velora-cognitive-voice.tar.gz`.

3. Download offline package wheels for future pip-only installation (optional):

```bash
cd ~/workspace/nasw
mkdir -p offline_packages
python -m pip download -r requirements-server.txt -d offline_packages
```

4. Run a smoke test using the sample dataset:

```bash
cd ~/workspace/nasw
python 11-create_sample_dataset.py --overwrite
python 10-train_from_raw_dataset.py --overwrite --tasks ALL --weights none --epochs 1 --batch-size 4
```

5. If this passes, copy the repository, `download/velora-cognitive-voice.tar.gz`,
and optionally the `offline_packages/` directory to the closed GPU server before
the network is cut off.

If the real `dataset/` folder arrives later on the closed server, the same
pipeline can be run there using the prepared environment and package cache.

## Closed GPU Server Preflight Checklist

Use this checklist before the AI Hub GPU server is isolated from the network.
The goal is to confirm that training can run without downloading anything after
the server becomes closed-network.

1. Keep private data and generated artifacts out of GitHub:

```text
dataset/
nasw/download/
offline_packages/
*.h5
*.tar.gz
```

The real AI Hub audio, generated spectrograms, trained model files, packaged
environments, and package caches should stay on the secured server or approved
transfer media only.

2. Prepare all offline dependencies before the network is closed:

```bash
cd ~/workspace/nasw
conda env create -f environment.yml
conda activate velora-cognitive-voice
conda install -c conda-forge conda-pack
mkdir -p download offline_packages
conda run -n velora-cognitive-voice conda-pack \
  -n velora-cognitive-voice \
  -o download/velora-cognitive-voice.tar.gz \
  --force
python -m pip download -r requirements-server.txt -d offline_packages
```

3. Download the VGG16 ImageNet weights if training will use
`--weights imagenet`:

```bash
python - <<'PY'
from tensorflow.keras.applications import VGG16
VGG16(weights="imagenet", include_top=False, input_shape=(100, 100, 3))
print("VGG16 ImageNet weights downloaded.")
PY
```

Copy this cached file to the closed server:

```text
~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
```

This file is not required for inference with an already trained `.h5` model.
It is required only when training or fine-tuning starts from ImageNet weights.

4. Verify TensorFlow GPU access before isolation:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

The TensorFlow command should print at least one GPU device. If it prints `[]`,
fix the Python/CUDA runtime environment before the server is closed.

5. Run a one-epoch smoke test before isolation:

```bash
cd ~/workspace/nasw
python 11-create_sample_dataset.py --overwrite
python 10-train_from_raw_dataset.py --overwrite --tasks ALL --weights none --epochs 1 --batch-size 4
```

This confirms imports, audio processing, dataset preparation, model creation,
and the training loop. It is not an accuracy test.

6. Create checksums for transferred offline files:

```bash
cd ~/workspace/nasw
sha256sum download/velora-cognitive-voice.tar.gz > SHA256SUMS.txt
find offline_packages -type f -print0 | xargs -0 sha256sum >> SHA256SUMS.txt
sha256sum ~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 >> SHA256SUMS.txt
```

After copying files to the closed server, run `sha256sum -c SHA256SUMS.txt`
from the same relative layout to verify that the transfer is intact.

7. Copy the minimum offline transfer bundle:

```text
workspace/nasw/
workspace/nasw/download/velora-cognitive-voice.tar.gz
workspace/nasw/offline_packages/
~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
workspace/dataset/
SHA256SUMS.txt
```

8. Limit model export after closed-network training:

```text
normal_mci_ad_task-ALL_best.h5
normal_mci_ad_task-ALL_metadata.json
training CSV logs, if approved
environment or package version records, if approved
```

Do not export raw audio, generated spectrogram images, `dataset/`, or logs that
contain personal information unless the security policy explicitly allows it.

## Real Dataset Checklist

Use this checklist after the real AI Hub dataset is installed and before the
network is closed.

Expected layout:

```text
~/workspace/
  dataset/
    IB-APPS/
    CERAD-K/
    SNSB-II/
  nasw/
    download/
```

If the delivered dataset is nested differently, for example:

```text
~/workspace/dataset/some_export_folder/Training/...
~/workspace/dataset/01.source/...
~/workspace/dataset/02.label/...
```

the preparation script still scans recursively. It first looks for
`IB-APPS`, `CERAD-K`, and `SNSB-II` folder names anywhere below the source
root. If those names are not found, it falls back to scanning all JSON files
below the source root. When JSON and audio are stored in separate folders, it
also builds a filename index for `.flac`, `.wav`, `.mp3`, and `.m4a` files and
tries to match the audio by filename.

Use the broadest folder that contains both labels and audio:

```bash
python 07-prepare_normal_mci_ad_dataset.py \
  --source-root ~/workspace/dataset \
  --manifest-only
```

If the dataset was placed somewhere else:

```bash
python 07-prepare_normal_mci_ad_dataset.py \
  --source-root /path/to/delivered_dataset_root \
  --manifest-only
```

The output prints `source_roots_scanned` and `json_roots_scanned`; check these
first when `read` or `written` is 0. To disable recursive audio filename
matching for a very large dataset:

```bash
python 07-prepare_normal_mci_ad_dataset.py \
  --source-root ~/workspace/dataset \
  --manifest-only \
  --no-recursive-audio-search
```

Activate the training environment:

```bash
cd ~/workspace/nasw
conda activate velora-cognitive-voice
```

Check GPU and audio dependencies:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
ffmpeg -version
python -c "import librosa, soundfile, matplotlib, pandas, PIL; print('packages ok')"
```

If `nvidia-smi` sees the GPU but TensorFlow prints `[]` with
`Cannot dlopen some GPU libraries` and `Skipping registering GPU devices`,
install the TensorFlow CUDA extra inside the active environment:

```bash
python -m pip install --upgrade "tensorflow[and-cuda]==2.15.1"
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Check raw data folders and representative files:

```bash
ls -la ~/workspace/dataset
find ~/workspace/dataset -maxdepth 2 -type d | head -40
find ~/workspace/dataset -name '*_R.flac' | head
find ~/workspace/dataset -name '*_R.json' | head
```

Run a metadata-only scan first. This does not create spectrogram images:

```bash
python 07-prepare_normal_mci_ad_dataset.py --manifest-only
```

Review the printed counts:

```text
read: should be greater than 0
written: should be greater than 0
skipped_unknown_label: should be checked if high
skipped_missing_audio: should be checked if high
json_errors: should be 0 or investigated
```

The manifest is written here:

```text
~/workspace/nasw/download/prepared_normal_mci_ad/dataset_manifest.csv
```

The preprocessing settings used for training data creation are written here:

```text
~/workspace/nasw/download/prepared_normal_mci_ad/preprocessing_config.json
```

This file records `sample_rate`, `seconds`, `dpi`, `audio_kind`, and split
ratios. The training script stores this information in model metadata, and the
inference script reuses it automatically.

If the manifest counts look valid, create the Mel spectrogram training dataset:

```bash
python 07-prepare_normal_mci_ad_dataset.py --overwrite
```

Split ratio constraints:

```text
0 < train_ratio < 1
0 <= val_ratio < 1
train_ratio + val_ratio < 1
```

Check generated class folders:

```bash
find ~/workspace/nasw/download/prepared_normal_mci_ad/multiclass/ALL -maxdepth 2 -type d | sort
```

Expected folders:

```text
train/AD
train/MCI
train/Normal
validation/AD
validation/MCI
validation/Normal
test/AD
test/MCI
test/Normal
```

Train the model. Use `imagenet` while the network is still open:

```bash
python 10-train_from_raw_dataset.py --overwrite --tasks ALL --weights imagenet
```

If the network is closed or ImageNet weights are unavailable:

```bash
python 10-train_from_raw_dataset.py --overwrite --tasks ALL --weights none
```

Check training outputs:

```bash
ls -lh ~/workspace/nasw/download/ad_mci_normal/task-ALL/
cat ~/workspace/nasw/download/ad_mci_normal/training_summary.json
```

Expected output files:

```text
normal_mci_ad_task-ALL_best.h5
normal_mci_ad_task-ALL_final.h5
normal_mci_ad_task-ALL_metadata.json
normal_mci_ad_task-ALL_<timestamp>.csv
```

Run one inference smoke test:

```bash
AUDIO=$(find ~/workspace/dataset -name '*_R.flac' | head -1)
python 09-infer_normal_mci_ad.py --audio-file "$AUDIO"
```

If this succeeds, the real-data training pipeline is ready.

## Long Training And Resume

For large raw datasets, split the workflow into preparation and training. This
prevents repeated Mel spectrogram generation when you need to restart training.

Prepare the image dataset once:

```bash
cd ~/workspace/nasw
python 07-prepare_normal_mci_ad_dataset.py --overwrite
```

Start training:

```bash
python 08-train_normal_mci_ad_vgg16.py --tasks ALL --weights imagenet
```

The trainer continuously saves:

```text
download/ad_mci_normal/task-ALL/normal_mci_ad_task-ALL_best.h5
download/ad_mci_normal/task-ALL/normal_mci_ad_task-ALL_final.h5
```

If training is interrupted, resume from the existing final checkpoint:

```bash
python 08-train_normal_mci_ad_vgg16.py --tasks ALL --resume --resume-from final
```

Resume from the best validation checkpoint instead:

```bash
python 08-train_normal_mci_ad_vgg16.py --tasks ALL --resume --resume-from best
```

You can also resume through the one-command wrapper without regenerating
spectrograms:

```bash
python 10-train_from_raw_dataset.py --skip-prepare --tasks ALL --resume --resume-from final
```

To start from a specific exported model:

```bash
python 08-train_normal_mci_ad_vgg16.py \
  --tasks ALL \
  --initial-model ~/workspace/nasw/download/ad_mci_normal/task-ALL/normal_mci_ad_task-ALL_best.h5
```

One-command preparation and training:

```bash
cd ~/workspace/nasw
python 10-train_from_raw_dataset.py --overwrite --tasks ALL
```

```bash
cd ~/workspace/nasw
python 07-prepare_normal_mci_ad_dataset.py --overwrite
```

By default, the script scans all three source groups: `IB-APPS`, `CERAD-K`, and `SNSB-II`.
It uses `raw-file` / `_R.flac` audio by default.

## Train Current Model

```bash
cd ~/workspace/nasw
python 08-train_normal_mci_ad_vgg16.py --tasks ALL
```

If the server cannot download ImageNet weights:

```bash
python 08-train_normal_mci_ad_vgg16.py \
  --tasks ALL \
  --weights none
```

The default trained model output is:

```text
~/workspace/nasw/download/ad_mci_normal/task-ALL/
```

The prepared image dataset is:

```text
~/workspace/nasw/download/prepared_normal_mci_ad/
```

## Inference

```bash
cd ~/workspace/nasw
python 09-infer_normal_mci_ad.py --audio-file ~/workspace/dataset/SNSB-II/<test_idx>/<test_idx>_R.flac
```

If `--model` is omitted, the script automatically finds the latest best model
under `download/ad_mci_normal`.

## Connect To Backend

```bash
export VELORA_COGNITIVE_MODEL_PATH=~/workspace/nasw/download/ad_mci_normal/task-ALL/normal_mci_ad_task-ALL_best.h5
export VELORA_COGNITIVE_METADATA_PATH=~/workspace/nasw/download/ad_mci_normal/task-ALL/normal_mci_ad_task-ALL_metadata.json
```
