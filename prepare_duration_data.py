"""Prepare training data for duration model."""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import textgrid
import torch
import torchaudio
from tqdm import tqdm
from vocos.feature_extractors import MelSpectrogramFeatures


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--wav-dir", type=str, required=True)
    parser.add_argument("--textgrid-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


FLAGS = parse_args()
# Create output directory if it doesn't exist
output_dir = Path(FLAGS.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Interval:
    start_sample: int  # Start position in audio samples
    end_sample: int  # End position in audio samples
    start_frame: int  # Start position in spectrogram frames
    end_frame: int  # End position in spectrogram frames
    phone: str  # Phone label for this interval


@dataclass
class TextGridData:
    intervals: List[Interval]


def read_textgrid(tg_path: Path) -> List[str]:
    # Read TextGrid file
    tg = textgrid.TextGrid.fromFile(tg_path)

    # Get the phones tier (index 1)
    phones_tier = tg[1]

    # Convert intervals to our dataclass
    intervals = []
    prev_end_frame = None
    data = []
    for interval in phones_tier.intervals:
        phone = "EMPTY" if interval.mark == "" else interval.mark.upper()
        duration = interval.maxTime - interval.minTime
        data.append((phone, duration))
    return data


def get_speaker_id(wav_path: Path) -> str:
    """Extract speaker ID from wav path.
    Args:
        wav_path: Path to wav file
    Returns:
        Speaker ID as string
    """
    # Get filename stem and split by underscore
    return str(wav_path.stem).split("_")[0]


def load_data(textgrid_path: Path, wav_path: Path) -> int:
    """Process data from textgrid and wav files and save to disk.

    Args:
        textgrid_path: Path to textgrid directory
        wav_path: Path to wav directory
    """
    # Create dictionaries mapping file stems to full paths
    wav_files_dict = {wav_file.stem: wav_file for wav_file in wav_path.rglob("*.wav")}
    textgrid_files_dict = {
        tg_file.stem: tg_file for tg_file in textgrid_path.rglob("*.TextGrid")
    }
    # Print number of files
    print(f"Number of WAV files: {len(wav_files_dict)}")
    print(f"Number of TextGrid files: {len(textgrid_files_dict)}")
    # Print number of matching pairs
    matching_pairs = sum(1 for stem in wav_files_dict if stem in textgrid_files_dict)
    print(f"Number of matching pairs: {matching_pairs}")
    # Process all matching files
    data = []
    for file_stem in tqdm(wav_files_dict, desc="Processing files"):
        if file_stem not in textgrid_files_dict:
            continue

        try:
            wav_file = wav_files_dict[file_stem]
            textgrid_file = textgrid_files_dict[file_stem]
            speaker_id = get_speaker_id(wav_file)
            phones = read_textgrid(textgrid_file)
            data.append((speaker_id, phones))
        except Exception as e:
            print(f"Error processing {file_stem}: {str(e)}")
            continue

    return data


# Test the function
data = load_data(textgrid_path=Path(FLAGS.textgrid_dir), wav_path=Path(FLAGS.wav_dir))

# Get unique speaker IDs and phones from data
speaker_ids = set()
all_phones = set()

for speaker_id, phone_data in data:
    speaker_ids.add(speaker_id)
    for phone, duration in phone_data:
        all_phones.add(phone)

speaker_ids = sorted(list(speaker_ids))
all_phones = sorted(list(all_phones))

print("Unique speaker IDs:", speaker_ids)
print("\nUnique phones:", all_phones)

# Create mapping dictionaries
speaker_id_to_idx = {speaker_id: idx for idx, speaker_id in enumerate(speaker_ids)}
phone_to_idx = {phone: idx for idx, phone in enumerate(all_phones)}
# Add PAD token to phone_to_idx if not already present
if "PAD" not in phone_to_idx:
    phone_to_idx["PAD"] = len(phone_to_idx)

# Export mapping dictionaries to JSON
maps = {
    "speaker_id_to_idx": speaker_id_to_idx,
    "phone_to_idx": phone_to_idx,
}
with open(Path(FLAGS.output_dir) / "maps.json", "w") as f:
    json.dump(maps, f, indent=2)

# First pass - compute total length
total_length = 0
for speaker_id, phone_data in data:
    total_length += len(phone_data)

# Create memory mapped file
mmap_file = np.memmap(
    Path(FLAGS.output_dir) / "duration.npy",
    dtype=np.float16,
    mode="w+",
    shape=(total_length, 3),  # [duration, phone_idx, speaker_idx]
)

# Second pass - write data
current_idx = 0
# Initialize random number generator with seed
rng = np.random.Generator(np.random.PCG64(42))

# Shuffle the data
data = list(data)
rng.shuffle(data)


for speaker_id, phone_data in data:
    length = len(phone_data)

    # Convert phones and durations to arrays
    phones = np.array(
        [phone_to_idx[phone] for phone, _ in phone_data], dtype=np.float16
    )
    speaker_idx = np.full(length, speaker_id_to_idx[speaker_id], dtype=np.float16)
    durations = np.array([duration for _, duration in phone_data], dtype=np.float16)
    durations = np.clip(durations, 0, 1)

    # Write to memmap
    start_index = current_idx
    end_index = start_index + length
    mmap_file[start_index:end_index, 0] = durations
    mmap_file[start_index:end_index, 1] = phones
    mmap_file[start_index:end_index, 2] = speaker_idx
    current_idx = end_index

mmap_file.flush()
