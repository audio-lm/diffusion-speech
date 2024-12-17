# Diffusion Speech

Diffusion text to speech model.

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

Synthesize speech using pretrained models.

```
# download pretrained models
git clone https://huggingface.co/ntt123/diffusion-speech-360h /tmp/data

# download nltk data
uv run python -c "import nltk; nltk.download('cmudict')"
uv run python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

uv run synthesize.py \
--duration-model-config ./configs/train_duration_dit_s.yaml \
--acoustic-model-config ./configs/train_acoustic_dit_b.yaml \
--duration-model-checkpoint /tmp/data/duration_model_0120000.pt \
--acoustic-model-checkpoint /tmp/data/acoustic_model_0140000.pt \
--speaker-id 1914 \
--output-file ./audio.wav \
--text "Ilya has made several major contributions to the field of deep learning!"
```

See an example of the generated audio at [audio.wav](audio.wav).