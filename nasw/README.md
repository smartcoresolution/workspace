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
