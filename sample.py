# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from diffusion import create_diffusion
from models import DiT_models


def find_model(model_name):
    assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        print("Using EMA model")
        checkpoint = checkpoint["ema"]
    else:
        print("Using model")
        checkpoint = checkpoint["model"]
    return checkpoint


def get_batch(
    step, batch_size, seq_len, DEVICE, data_file, data_dim, data_mean, data_std
):
    # Load dataset from memmap file
    arr = np.memmap(data_file, dtype=np.float16, mode="r")
    arr = np.memmap(
        data_file,
        dtype=np.float16,
        mode="r",
        shape=(arr.shape[0] // (data_dim + 2), data_dim + 2),
    )

    # Create random number generator
    rng = np.random.Generator(np.random.PCG64(seed=step))

    # Generate start indices and convert to integer array
    start_indices = rng.choice(
        arr.shape[0] - seq_len, size=batch_size, replace=False
    ).astype(np.int64)

    # Create batch data array
    batch_data = np.zeros((batch_size, seq_len, data_dim + 2), dtype=np.float16)
    # Fill batch data one sequence at a time
    for i, start_idx in enumerate(start_indices):
        batch_data[i] = arr[start_idx : start_idx + seq_len]

    # Extract features
    x = batch_data[:, :, :data_dim].astype(np.float16)
    x = np.moveaxis(x, 1, 2)
    phone = batch_data[:, :, data_dim].astype(np.int32)
    speaker_id = batch_data[:, :, data_dim + 1].astype(np.int32)

    # convert to torch tensors
    x = torch.from_numpy(x).to(DEVICE)
    x = (x - data_mean) / data_std
    phone = torch.from_numpy(phone).to(DEVICE)
    speaker_id = torch.from_numpy(speaker_id).to(DEVICE)

    return x, speaker_id, phone


def get_data(config_path, seed=0):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    model_config = config["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, speaker_id, phone = get_batch(
        seed,
        1,
        seq_len=model_config["input_size"],
        DEVICE=device,
        data_file=data_config["data_path"],
        data_dim=data_config["data_dim"],
        data_mean=data_config["data_mean"],
        data_std=data_config["data_std"],
    )

    return x, speaker_id, phone


def plot_samples(samples, x):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 4))
    plt.tight_layout()

    # Function to update frame
    def update(frame):
        ax.clear()
        ax.text(
            0.02,
            0.98,
            f"{frame+1} / 1000",
            transform=ax.transAxes,
            verticalalignment="top",
            color="black",
        )
        if samples[frame].shape[1] > 1:
            im = ax.imshow(
                samples[frame].cpu().numpy()[0],
                origin="lower",
                aspect="auto",
                interpolation="none",
                vmin=-5,
                vmax=5,
            )
            return [im]
        elif samples[frame].shape[1] == 1:
            line1 = ax.plot(samples[frame].cpu().numpy()[0, 0])[0]
            line2 = ax.plot(x.cpu().numpy()[0, 0])[0]
            plt.ylim(-10, 10)
            return [line1, line2]

    from tqdm import tqdm

    # Create animation with progress bar
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(range(len(samples)), desc="Generating animation"),
        interval=1000 / 60,
        blit=True,  # 24 fps
    )

    # Save as MP4
    anim.save("animation.mp4", fps=60, extra_args=["-vcodec", "libx264"])
    plt.close()


def sample(
    config_path,
    ckpt_path,
    cfg_scale=4.0,
    num_sampling_steps=1000,
    seed=0,
    speaker_id=None,
    phone=None,
):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    model_config = config["model"]

    # Load model:
    model = DiT_models[model_config["name"]](
        input_size=model_config["input_size"],
        embedding_vocab_size=model_config["embedding_vocab_size"],
        learn_sigma=model_config["learn_sigma"],
        in_channels=data_config["data_dim"],
    ).to(device)

    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(num_sampling_steps))
    n = 1
    z = torch.randn(n, data_config["data_dim"], speaker_id.shape[1], device=device)

    attn_mask = speaker_id[:, None, :] == speaker_id[:, :, None]
    attn_mask = attn_mask.unsqueeze(1)
    attn_mask = torch.cat([attn_mask, attn_mask], 0)
    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    unconditional_value = model.y_embedder.unconditional_value
    phone_null = torch.full_like(phone, unconditional_value)
    speaker_id_null = torch.full_like(speaker_id, unconditional_value)
    phone = torch.cat([phone, phone_null], 0)
    speaker_id = torch.cat([speaker_id, speaker_id_null], 0)
    model_kwargs = dict(
        phone=phone,
        speaker_id=speaker_id,
        cfg_scale=cfg_scale,
        attn_mask=attn_mask,
    )

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples = [s.chunk(2, dim=0)[0] for s in samples]  # Remove null class samples
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    x, speaker_id, phone = get_data(args.config, args.seed)
    samples = sample(
        args.config,
        args.ckpt,
        args.cfg_scale,
        args.num_sampling_steps,
        args.seed,
        speaker_id,
        phone,
    )
    plot_samples(samples, x)
