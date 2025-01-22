import argparse
import glob
import io
import json
import os
import random
import shutil
import zipfile
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare duration data from MLS dataset"
    )
    parser.add_argument(
        "--speaker-config",
        type=str,
        default="mls_dataset/speaker_id_to_idx.json",
        help="Path to speaker ID to index mapping config",
    )
    parser.add_argument(
        "--phone-config",
        type=str,
        default="mls_dataset/phone_to_idx.json",
        help="Path to phone to index mapping config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/duration",
        help="Output directory for duration data",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of worker processes"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for train/test split"
    )
    return parser.parse_args()


phone_kind_to_idx = {
    "SIL": 0,
    "WORD": 1,
    "START": 2,
    "END": 3,
    "MIDDLE": 4,
}


def load_textgrid(s: str):
    lines = s.splitlines()
    lines = [line.strip() for line in lines]
    num_word_intervals = int(lines[13][18:])
    word_intervals = []
    for i in range(num_word_intervals):
        base = 14 + i * 4
        xmin = float(lines[base + 1][7:])
        xmax = float(lines[base + 2][7:])
        text = lines[base + 3][8:-1]
        word_intervals.append((xmin, xmax, text))

    num_phone_intervals = int(lines[14 + num_word_intervals * 4 + 5][18:])
    phone_intervals = []
    for i in range(num_phone_intervals):
        j = 20 + num_word_intervals * 4
        base = j + i * 4
        xmin = float(lines[base + 1][7:])
        xmax = float(lines[base + 2][7:])
        text = lines[base + 3][8:-1]
        phone_intervals.append((xmin, xmax, text))

    return word_intervals, phone_intervals


def prepare_data(
    word_intervals, phone_intervals, speaker_id_idx, phone_to_idx: Dict
) -> List[str]:
    data = []
    current_word_idx = 0

    for phone_xmin, phone_xmax, phone_text in phone_intervals:
        phone = "SIL" if phone_text == "" else phone_text.upper()
        duration = phone_xmax - phone_xmin

        if phone == "SIL":
            phone_kind = "SIL"
            current_word_idx += 1
        elif (
            phone_xmin == word_intervals[current_word_idx][0]
            and phone_xmax == word_intervals[current_word_idx][1]
        ):
            phone_kind = "WORD"
            current_word_idx += 1
        elif phone_xmin == word_intervals[current_word_idx][0]:
            phone_kind = "START"
        elif phone_xmax == word_intervals[current_word_idx][1]:
            phone_kind = "END"
            current_word_idx += 1
        else:
            phone_kind = "MIDDLE"

        # convert to index
        phone_idx = phone_to_idx[phone]
        phone_kind_idx = phone_kind_to_idx[phone_kind]
        data.append((duration, phone_idx, speaker_id_idx, phone_kind_idx))
    return data


def download_and_unzip(
    args, speaker_id_to_idx: Dict, phone_to_idx: Dict, file_num: int, total_entries: int
):
    # Generate file number with leading zeros
    file_num_str = str(file_num).zfill(5)
    total_entries_str = str(total_entries).zfill(5)

    # Download the zip file
    url = f"https://huggingface.co/datasets/ntt123/aligned_mls_eng/resolve/main/data/train-{file_num_str}-of-{total_entries_str}.zip"

    try:
        # Download to memory
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Read zip file in memory
        zip_data = io.BytesIO(response.content)

        # Process zip contents in memory
        with zipfile.ZipFile(zip_data) as zip_ref:
            # Process TextGrid files
            for filename in zip_ref.namelist():
                speaker_id = filename.split("/")[0]
                speaker_id_idx = speaker_id_to_idx[speaker_id]
                if filename.endswith(".TextGrid"):
                    # Read TextGrid file content directly from zip
                    with zip_ref.open(filename) as f:
                        content = f.read().decode("utf-8")
                        word_intervals, phone_intervals = load_textgrid(content)
                        data = prepare_data(
                            word_intervals,
                            phone_intervals,
                            speaker_id_idx,
                            phone_to_idx,
                        )
                        data = np.array(data, dtype=np.float16)
                        # clip duration to [0, 1]
                        data[:, 0] = np.clip(data[:, 0], 0, 1)
                        # get file stem from filename
                        file_stem = Path(filename).stem

                        np.save(f"{args.output_dir}/files/{file_stem}.npy", data)

    except Exception as e:
        print(f"Error processing file {file_num_str}: {str(e)}")


def save_to_disk(args, files, output_split):
    # write file list to file
    with open(f"{args.output_dir}/{output_split}.txt", "w") as f:
        for file in files:
            f.write(file + "\n")
    current_idx = 0
    # Create memory mapped file
    mmap_file = f"{args.output_dir}/{output_split}.bin"
    # Initial size estimate (can be adjusted)
    initial_size = 5_000_000_000  # Large enough to hold all data
    mmap_array = np.memmap(
        mmap_file, dtype=np.float16, mode="w+", shape=(initial_size, 4)
    )
    for file in tqdm(files, desc="Writing to memmap"):
        data = np.load(file)
        array_length = len(data)
        mmap_array[current_idx : current_idx + array_length] = data
        current_idx += array_length
    # Flush and close the original memmap
    mmap_array.flush()
    del mmap_array

    # Resize the file to the actual data size
    new_size = (
        current_idx * 4 * np.dtype(np.float16).itemsize
    )  # Calculate actual bytes needed
    with open(mmap_file, "r+b") as f:
        f.truncate(new_size)

    # Create new memmap with correct size
    final_shape = (current_idx, 4)
    final_array = np.memmap(mmap_file, dtype=np.float16, mode="r+", shape=final_shape)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load speaker and phone mappings
    with open(args.speaker_config) as f:
        speaker_id_to_idx = json.load(f)

    with open(args.phone_config) as f:
        phone_to_idx = json.load(f)

    maps = {
        "speaker_id_to_idx": speaker_id_to_idx,
        "phone_to_idx": phone_to_idx,
        "phone_kind_to_idx": phone_kind_to_idx,
    }
    with open(os.path.join(args.output_dir, "maps.json"), "w") as f:
        json.dump(maps, f, indent=2)

    os.makedirs(f"{args.output_dir}/files", exist_ok=True)

    total_entries = 1416
    file_nums = range(total_entries)

    # with tqdm(total=len(file_nums)) as pbar:

    #     def update(*a):
    #         pbar.update()

    #     # Create process pool
    #     with Pool(processes=args.num_workers) as pool:
    #         # Map file numbers to worker processes
    #         results = []
    #         for file_num in file_nums:
    #             result = pool.apply_async(
    #                 download_and_unzip,
    #                 (args, speaker_id_to_idx, phone_to_idx, file_num, total_entries),
    #                 callback=update,
    #             )
    #             results.append(result)

    #         # Get results and write to memmap
    #         for result in results:
    #             result.get()

    # read all files in data/duration/{output_split}
    files = sorted(glob.glob(f"{args.output_dir}/files/*.npy"))
    random.Random(args.random_seed).shuffle(files)
    N = len(files)
    L = N * 9 // 10
    train_files = files[:L]
    test_files = files[L:]
    save_to_disk(args, train_files, "train")
    save_to_disk(args, test_files, "test")


if __name__ == "__main__":
    main()
