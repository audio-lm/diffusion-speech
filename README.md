# Diffusion Speech

Diffusion Speech is a diffusion-based text-to-speech model. Our speech synthesis pipeline is quite simple. We use a diffusion transformer model (DiT) to predict the duration of each phoneme. Then we use another DiT model to predict the mel-spectrogram features at each frame. Finally, we use the pretrained [Vocos vocoder](https://github.com/gemelo-ai/vocos) to convert the mel-spectrogram to audio waveform.


To get started, first install the required system dependencies and Python package manager uv:

```
apt update
apt install ffmpeg git-lfs zip unzip -y
git lfs install

pip install uv
uv venv -p python3.11
uv pip install -r pyproject.toml
```

## Dataset

We use the LibriTTS-R dataset with phoneme alignment ground truth provided by [cdminix](https://huggingface.co/datasets/cdminix/libritts-r-aligned) for training duration and acoustic models.

```
(
    mkdir -p /tmp/data
    cd /tmp/data
    git clone https://huggingface.co/datasets/cdminix/libritts-r-aligned
    ( cd libritts-r-aligned/data; tar --no-same-owner -xzf train_clean_360.tar.gz )
    wget https://us.openslr.org/resources/141/train_clean_360.tar.gz
    tar --no-same-owner -xzf train_clean_360.tar.gz
)
```

## Duration model
Prepare the training data for duration model:

```
uv run prepare_duration_data.py \
--wav-dir /tmp/data/LibriTTS_R/train-clean-360 \
--textgrid-dir /tmp/data/libritts-r-aligned/data \
--output-dir /tmp/data
```


Start training the duration model:

```
uv run train.py --config configs/train_duration_dit_s.yaml
```

Sample from the duration model:

```
uv run sample.py \
--config configs/train_duration_dit_s.yaml \
--ckpt results/duration/000-DiT-S/checkpoints/0100000.pt \
--cfg-scale 4 \
--num-sampling-steps 1000
```

## Acoustic model

Prepare the training data for acoustic model:

```
uv run prepare_acoustic_data.py \
--wav-dir /tmp/data/LibriTTS_R/train-clean-360 \
--textgrid-dir /tmp/data/libritts-r-aligned/data \
--output-dir /tmp/data
```


Start training the acoustic model:

```
uv run train.py --config configs/train_acoustic_dit_b.yaml
```

Sample from the acoustic model:

```
uv run sample.py \
--config configs/train_acoustic_dit_b.yaml \
--ckpt results/acoustic/000-DiT-B/checkpoints/0100000.pt \
--cfg-scale 4 \
--num-sampling-steps 1000
```

## Pretrained models

To synthesize speech using pretrained models:

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

# Call for Compute Donations

The current model is trained on only 360 hours of speech data. I would love to train even bigger and better open-source TTS models on much more data (10x, 100x, or even 1000x more!), but I'm currently limited by compute resources. If you're interested in supporting this project by donating compute resources, I'd be very grateful! You can reach me at `xcodevn@gmail.com`.
