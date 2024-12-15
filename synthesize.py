"""
Synthesize a given text using the trained DiT models.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import yaml
from g2p_en import G2p

from sample import sample

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, required=True)
parser.add_argument("--duration-model-config", type=str, required=True)
parser.add_argument("--duration-model-checkpoint", type=str, required=True)
parser.add_argument("--acoustic-model-config", type=str, required=True)
parser.add_argument("--acoustic-model-checkpoint", type=str, required=True)
parser.add_argument("--output-file", type=str, required=True)
parser.add_argument("--speaker-id", type=int, required=True)


args = parser.parse_args()

print("Text:", args.text)

# Read duration model config
with open(args.duration_model_config, "r") as f:
    duration_config = yaml.safe_load(f)

# Get data directory from data_path
data_dir = os.path.dirname(duration_config["data"]["data_path"])

# Read maps.json from same directory
with open(os.path.join(data_dir, "maps.json"), "r") as f:
    maps = json.load(f)
phone_to_idx = maps["phone_to_idx"]


# Step 1: Text to phonemes
def text_to_phonemes(text, insert_empty=True):
    g2p = G2p()
    phonemes = g2p(text)
    output = []
    for p in phonemes:
        if p in phone_to_idx:
            pass
        elif p in [" ", ",", ".", "!", "?", ";", ":"]:
            p = "EMPTY"
        else:
            continue
        # remove repeated phonemes
        if len(output) > 0 and output[-1] == p:
            continue
        output.append(p)

    if insert_empty:
        if output[0] != "EMPTY":
            output.insert(0, "EMPTY")
        if output[-1] != "EMPTY":
            output.append("EMPTY")
    return output


phonemes = text_to_phonemes(args.text)
# Convert phonemes to indices
phoneme_indices = [phone_to_idx[p] for p in phonemes]

print("Phonemes:", phonemes)

# Step 2: Duration prediction

# conver phoneme_indices to torch tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_phoneme_indices = torch.tensor(phoneme_indices)[None, :].long().to(device)
torch_speaker_id = torch.full_like(torch_phoneme_indices, args.speaker_id)

samples = sample(
    args.duration_model_config,
    args.duration_model_checkpoint,
    cfg_scale=2.0,
    num_sampling_steps=1000,
    seed=0,
    speaker_id=torch_speaker_id,
    phone=torch_phoneme_indices,
)
phoneme_durations = samples[-1][0, 0]
# plot phoneme durations and save to file
plt.figure(figsize=(12, 4))
durations = phoneme_durations.cpu().numpy()
plt.plot(durations)
plt.xticks(range(len(phonemes)), phonemes, rotation=45)
plt.grid(True)
plt.title("Phoneme Durations")
plt.xlabel("Phoneme")
plt.ylabel("Duration")
plt.tight_layout()
plt.savefig("phoneme_durations.png")
plt.close()

# Step 3: Acoustic prediction
# First, we need to convert phoneme durations to number of frames per phoneme (min 1 frame)
SAMPLE_RATE = 24000
HOP_LENGTH = 256
N_FFT = 1024
N_MELS = 100
time_per_frame = HOP_LENGTH / SAMPLE_RATE
# convert predicted durations to raw durations using data mean and std in the config
if duration_config["data"]["normalize"]:
    mean = duration_config["data"]["data_mean"]
    std = duration_config["data"]["data_std"]
    raw_durations = phoneme_durations * std + mean
else:
    raw_durations = phoneme_durations

raw_durations = raw_durations.clamp(min=time_per_frame, max=1.0)
end_time = torch.cumsum(raw_durations, dim=0)
end_frame = end_time / time_per_frame
int_end_frame = end_frame.floor().int()
repeated_phoneme_indices = []
for i in range(len(phonemes)):
    repeated_phoneme_indices.extend(
        [phoneme_indices[i]] * (int_end_frame[i] - len(repeated_phoneme_indices))
    )

torch_phoneme_indices = (
    torch.tensor(repeated_phoneme_indices)[None, :].long().to(device)
)
torch_speaker_id = torch.full_like(torch_phoneme_indices, args.speaker_id)

samples = sample(
    args.acoustic_model_config,
    args.acoustic_model_checkpoint,
    cfg_scale=2.0,
    num_sampling_steps=1000,
    seed=0,
    speaker_id=torch_speaker_id,
    phone=torch_phoneme_indices,
)
mel = samples[-1][0]
# compute raw mel if acoustic model normalize is true
acoustic_config = yaml.safe_load(open(args.acoustic_model_config, "r"))
if acoustic_config["data"]["normalize"]:
    mean = acoustic_config["data"]["data_mean"]
    std = acoustic_config["data"]["data_std"]
    raw_mel = mel * std + mean
else:
    raw_mel = mel


# plot melspectrogram
plt.figure(figsize=(12, 4))
plt.imshow(raw_mel.cpu().numpy(), aspect="auto", origin="lower")
plt.colorbar()
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.tight_layout()
plt.savefig("mel_spectrogram.png")
plt.close()

# Step 4: Vocoder

from vocos import Vocos

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")


audio = vocos.decode(raw_mel.cpu()[None, :, :]).squeeze().cpu().numpy()
import soundfile as sf

sf.write(args.output_file, audio, 24000)
