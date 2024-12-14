# Diffusion Speech

A text to speech model using diffusion models.

To get started, you need to install the dependencies and download the dataset. We use uv for managing Python dependencies.

```
pip install uv
uv venv -p python3.11
uv pip install -r pyproject.toml
```

We use LibriTTS-R dataset.

```
apt update
apt install ffmpeg git-lfs zip unzip -y
git lfs install

(
    mkdir -p /tmp/data
    cd /tmp/data
    git clone https://huggingface.co/datasets/cdminix/libritts-r-aligned
    ( cd libritts-r-aligned/data; tar --no-same-owner -xzf train_clean_360.tar.gz )
    wget https://us.openslr.org/resources/141/train_clean_360.tar.gz
    tar --no-same-owner -xzf train_clean_360.tar.gz
)
```


Prepare the training data for duration model.

```
uv run prepare_duration_data.py \
--wav-dir /tmp/data/LibriTTS_R/train-clean-360 \
--textgrid-dir /tmp/data/libritts-r-aligned/data \
--output-dir /tmp/data
```


Start training the duration model.

```
uv run train.py --config configs/train_duration_dit_s.yaml
```

Sample from the duration model.

```
uv run sample.py \
--config configs/train_duration_dit_s.yaml \
--ckpt results/duration/000-DiT-S/checkpoints/0100000.pt \
--cfg-scale 4 \
--num-sampling-steps 1000
```

Prepare the training data for acoustic model.

```
uv run prepare_acoustic_data.py \
--wav-dir /tmp/data/LibriTTS_R/train-clean-360 \
--textgrid-dir /tmp/data/libritts-r-aligned/data \
--output-dir /tmp/data
```


Start training the acoustic model.

```
uv run train.py --config configs/train_acoustic_dit_b.yaml
```

Sample from the acoustic model.

```
uv run sample.py \
--config configs/train_acoustic_dit_b.yaml \
--ckpt results/acoustic/000-DiT-B/checkpoints/0100000.pt \
--cfg-scale 4 \
--num-sampling-steps 1000
```
