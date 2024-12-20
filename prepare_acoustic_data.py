import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import textgrid
import torch
import torchaudio
from tqdm import tqdm
from vocos.feature_extractors import MelSpectrogramFeatures


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare acoustic training data")
    parser.add_argument(
        "--wav-dir", type=str, required=True, help="Directory containing WAV files"
    )
    parser.add_argument(
        "--textgrid-dir",
        type=str,
        required=True,
        help="Directory containing TextGrid files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
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
    phone_kind: str  # Phone kind for this interval


@dataclass
class TextGridData:
    intervals: List[Interval]

    def padded_phones(self, L: int) -> Tuple[List[str], List[str]]:
        """Return a list of phones repeated for each frame and padded/truncated to length L.

        Each phone is repeated for the number of frames it spans in the interval.
        If the resulting sequence is shorter than L, it is padded with 'PAD' tokens.
        If longer than L, it is truncated to length L.

        Args:
            L: Target length for the output sequence

        Returns:
            Tuple containing:
            - List of phone labels of length L, with each phone repeated for its frame duration
            - List of phone kinds of length L
        """
        phones = []
        phone_kinds = []
        for interval in self.intervals:
            # Repeat phone for each frame in the interval
            cur_frame = len(phones)
            phones.extend([interval.phone] * (interval.end_frame - cur_frame))
            phone_kinds.extend([interval.phone_kind] * (interval.end_frame - cur_frame))
        # Pad with empty tokens if needed
        if len(phones) < L:
            phones.extend(["PAD"] * (L - len(phones)))
            phone_kinds.extend(["EMPTY"] * (L - len(phone_kinds)))
        elif len(phones) > L:
            phones = phones[:L]
            phone_kinds = phone_kinds[:L]

        return phones, phone_kinds


def read_textgrid(
    tg_path: Path, sample_rate: int, hop_length: int, pad_length: int
) -> Tuple[List[str], List[str]]:
    """Read a TextGrid file and convert it to a sequence of frame-aligned phone labels.

    Args:
        tg_path: Path to TextGrid file
        sample_rate: Audio sample rate in Hz
        hop_length: Number of audio samples between consecutive frames
        pad_length: Target length for output sequence (will pad/truncate to this length)

    Returns:
        Tuple containing:
        - List of phone labels of length pad_length
        - List of phone kinds of length pad_length

    Raises:
        ValueError: If there is a discontinuity between intervals in the frame sequence
    """
    # Read TextGrid file
    tg = textgrid.TextGrid.fromFile(tg_path)

    words_tier = tg[0]
    phones_tier = tg[1]

    # Convert intervals to our dataclass
    intervals = []
    prev_end_frame = None
    current_word_idx = 0
    for interval in phones_tier.intervals:
        phone = "EMPTY" if interval.mark == "" else interval.mark.upper()
        start_sample = int(interval.minTime * sample_rate)
        end_sample = int(interval.maxTime * sample_rate)
        start_frame = start_sample // hop_length
        end_frame = end_sample // hop_length

        # Verify frame continuity
        if prev_end_frame is not None and start_frame != prev_end_frame:
            raise ValueError(
                f"Frame discontinuity detected: previous end_frame {prev_end_frame} != current start_frame {start_frame}"
            )

        if phone == "EMPTY":
            phone_kind = "EMPTY"
            current_word_idx += 1
        elif (
            interval.minTime == words_tier.intervals[current_word_idx].minTime
            and interval.maxTime == words_tier.intervals[current_word_idx].maxTime
        ):
            phone_kind = "WORD"
            current_word_idx += 1
        elif interval.minTime == words_tier.intervals[current_word_idx].minTime:
            phone_kind = "START"
        elif interval.maxTime == words_tier.intervals[current_word_idx].maxTime:
            phone_kind = "END"
            current_word_idx += 1
        else:
            phone_kind = "MIDDLE"

        intervals.append(
            Interval(
                start_sample=start_sample,
                end_sample=end_sample,
                start_frame=start_frame,
                end_frame=end_frame,
                phone=phone,
                phone_kind=phone_kind,
            )
        )
        prev_end_frame = end_frame
    return TextGridData(intervals=intervals).padded_phones(pad_length)


def get_speaker_id(wav_path: Path) -> str:
    """Extract speaker ID from wav path.
    Args:
        wav_path: Path to wav file
    Returns:
        Speaker ID as string
    """
    # Get filename stem and split by underscore
    return str(wav_path.stem).split("_")[0]


def load_data(
    textgrid_path: Path,
    wav_path: Path,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: int,
) -> int:
    """Process data from textgrid and wav files and save to disk.

    Args:
        textgrid_path: Path to textgrid directory
        wav_path: Path to wav directory
        sample_rate: Audio sample rate in Hz
        hop_length: Number of samples between successive frames
        n_fft: Size of FFT
        n_mels: Number of mel filterbanks

    Returns:
        Total length of all mel spectrograms processed
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

    # Initialize mel feature extractor
    mel_feature_extractor = MelSpectrogramFeatures(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        padding="same",
    ).cuda()

    mel_dir = output_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)

    total_mel_length = 0

    # Process all matching files
    for file_stem in tqdm(wav_files_dict, desc="Processing files"):
        if file_stem not in textgrid_files_dict:
            continue

        try:
            wav_file = wav_files_dict[file_stem]
            textgrid_file = textgrid_files_dict[file_stem]
            speaker_id = get_speaker_id(wav_file)

            # Load and process audio
            audio, sr = torchaudio.load(str(wav_file))
            audio = audio.cuda()

            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate).cuda()
                audio = resampler(audio)

            mel = mel_feature_extractor(audio)
            total_mel_length += mel.shape[-1]
            # Get phones
            phones, phone_kinds = read_textgrid(
                textgrid_file,
                sample_rate=sample_rate,
                hop_length=hop_length,
                pad_length=mel.shape[-1],
            )

            # Save features to disk
            output_path = mel_dir / f"{file_stem}.npy"
            np.save(
                output_path,
                {
                    "speaker_id": speaker_id,
                    "mel": mel.half().cpu().numpy(),
                    "phones": phones,
                    "phone_kinds": phone_kinds,
                },
            )
        except Exception as e:
            print(f"Error processing {file_stem}: {str(e)}")
            continue

    return total_mel_length


SAMPLE_RATE = 24000
HOP_LENGTH = 256
N_FFT = 1024
N_MELS = 100


# Test the function
total_mel_length = load_data(
    textgrid_path=Path(FLAGS.textgrid_dir),
    wav_path=Path(FLAGS.wav_dir),
    sample_rate=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
    n_fft=N_FFT,
    n_mels=N_MELS,
)

# Load mapping dictionaries from JSON
maps_path = Path(FLAGS.output_dir) / "maps.json"
if not maps_path.exists():
    raise FileNotFoundError(
        f"maps.json not found at {maps_path}. Please run prepare_duration_data.py first to generate the mapping files."
    )

with open(maps_path, "r") as f:
    maps = json.load(f)
    speaker_id_to_idx = maps["speaker_id_to_idx"]
    phone_to_idx = maps["phone_to_idx"]
    phone_kind_to_idx = maps["phone_kind_to_idx"]

# Create memory mapped file
D = N_MELS
mmap_file = np.memmap(
    Path(FLAGS.output_dir) / "acoustic.npy",
    dtype=np.float16,
    mode="w+",
    shape=(total_mel_length, D + 3),
)

# Write data
current_idx = 0
mel_dir = Path(FLAGS.output_dir) / "mel"
npy_files = sorted(mel_dir.glob("*.npy"))
rng = np.random.Generator(np.random.PCG64(42))
rng.shuffle(npy_files)

for npy_file in npy_files:
    record = np.load(npy_file, allow_pickle=True, mmap_mode=None).item()
    phones = [phone_to_idx[phone] for phone in record["phones"]]
    length = len(phones)
    phones = phones[:length]
    speaker_id = speaker_id_to_idx[record["speaker_id"]]
    phone_kinds = [
        phone_kind_to_idx[phone_kind] for phone_kind in record["phone_kinds"]
    ]
    phone_kinds = phone_kinds[:length]
    mel = record["mel"][0, :, :length].T

    # Write to memmap
    start_index = current_idx
    end_index = start_index + length
    mmap_file[start_index:end_index, :D] = mel
    mmap_file[start_index:end_index, D] = phones
    mmap_file[start_index:end_index, D + 1] = speaker_id
    mmap_file[start_index:end_index, D + 2] = phone_kinds
    current_idx = end_index

mmap_file.flush()
del mmap_file

# remove mel directory and all files in it
shutil.rmtree(mel_dir)
