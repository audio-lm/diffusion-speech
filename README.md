# Diffusion Speech

A text to speech model using diffusion models.

To get started, you need to install the dependencies and download the dataset. We use uv for managing Python dependencies.

```
pip install uv
export UV_PROJECT_ENVIRONMENT=/tmp/uv_venv
uv venv
source /tmp/uv_venv/bin/activate
# uv add torch vocos matplotlib textgrid wandb numpy
uv sync
wandb login
```

We use LibriTTS-R dataset.

```
mkdir -p /tmp/data
# install ffmpeg, git-lfs, zip, unzip
apt update; apt install ffmpeg git-lfs zip unzip -y
git lfs install

(
    cd /tmp/data
    git clone https://huggingface.co/datasets/cdminix/libritts-r-aligned
    ( cd libritts-r-aligned/data; tar --no-same-owner -xzf train_clean_360.tar.gz )
    wget https://us.openslr.org/resources/141/train_clean_360.tar.gz; tar --no-same-owner -xzf train_clean_360.tar.gz
)
```


Prepare the training data for acoustic model.

```
uv run prepare_acoustic_data.py \
--wav-dir /tmp/data/LibriTTS_R/train-clean-360 \
--textgrid-dir /tmp/data/libritts-r-aligned/data \
--output-dir /tmp/data/acoustic
```


Start training the acoustic model.

```
uv run train.py --config configs/train_acoustic_dit_b.yaml
```
