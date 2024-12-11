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

import numpy as np
import soundfile as sf
import torch
from vocos import Vocos

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


def get_batch(step, batch_size, seq_len, DEVICE, data_file):
    # Load dataset from memmap file
    arr = np.memmap(data_file, dtype=np.float16, mode="r")
    arr = np.memmap(
        data_file,
        dtype=np.float16,
        mode="r",
        shape=(arr.shape[0] // 102, 102),
    )

    # Create random number generator
    rng = np.random.Generator(np.random.PCG64(seed=step))

    # Generate start indices and convert to integer array
    start_indices = rng.choice(
        arr.shape[0] - seq_len, size=batch_size, replace=False
    ).astype(np.int64)

    # Create batch data array
    batch_data = np.zeros((batch_size, seq_len, 102), dtype=np.float16)

    # Fill batch data one sequence at a time
    for i, start_idx in enumerate(start_indices):
        batch_data[i] = arr[start_idx : start_idx + seq_len]

    # Extract features
    mel = batch_data[:, :, :100].astype(np.float16)
    mel = np.moveaxis(mel, 1, 2)
    phone = batch_data[:, :, 100].astype(np.int32)
    speaker_id = batch_data[:, :, 101].astype(np.int32)

    # convert to torch tensors
    mel = torch.from_numpy(mel).to(DEVICE)
    phone = torch.from_numpy(phone).to(DEVICE)
    speaker_id = torch.from_numpy(speaker_id).to(DEVICE)

    return mel, (speaker_id, phone)


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert (
            args.model == "DiT-XL/2"
        ), "Only DiT-XL/2 models are available for auto-download."

    # Load model:
    model = DiT_models[args.model](
        input_size=args.input_size, in_channels=args.in_channels
    ).to(device)

    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    n = 1
    z = torch.randn(n, args.in_channels, args.input_size, device=device)

    mel, (speaker_id, phone) = get_batch(
        args.seed,
        1,
        seq_len=args.input_size,
        DEVICE=device,
        data_file=args.data_file,
    )

    samples = mel.to(device).float()

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    phone_null = phone.clone()
    speaker_id_null = speaker_id.clone()
    phone_null.fill_(1024)
    speaker_id_null.fill_(1024)
    phone = torch.cat([phone, phone_null], 0)
    speaker_id = torch.cat([speaker_id, speaker_id_null], 0)
    model_kwargs = dict(phone=phone, speaker_id=speaker_id, cfg_scale=args.cfg_scale)

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

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    audio = vocos.decode(torch.clamp(samples[-1], -1, 1).cpu()).squeeze().cpu().numpy()

    sf.write("sample.wav", audio, 24000)

    # import matplotlib.animation as animation
    # import matplotlib.pyplot as plt

    # # Create figure and axis
    # fig, ax = plt.subplots(figsize=(20, 4))
    # plt.tight_layout()

    # # Convert samples to numpy array
    # mel_frames = samples

    # # Function to update frame
    # def update(frame):
    #     ax.clear()
    #     im = ax.imshow(
    #         mel_frames[frame].cpu().numpy()[0]*5,
    #         origin="lower",
    #         aspect="auto",
    #         interpolation="none",
    #         vmin=-5,
    #         vmax=5,
    #     )
    #     # if frame == 0:  # Only add colorbar on first frame
    #     #     plt.colorbar(im)
    #     ax.text(
    #         0.02,
    #         0.98,
    #         f"{frame+1} / 1000",
    #         transform=ax.transAxes,
    #         verticalalignment="top",
    #         color="black",
    #     )
    #     return [im]

    # # Create animation
    # anim = animation.FuncAnimation(
    #     fig, update, frames=len(mel_frames), interval=1000 / 60, blit=True  # 24 fps
    # )

    # # Save as MP4
    # anim.save("mel_animation.mp4", fps=60, extra_args=["-vcodec", "libx264"])
    # plt.close()
    # # with open("sample.dac", "wb") as f:
    # #     np.save(f, audio)
    # # Save and display images:
    # # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-B"
    )
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--in-channels", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-file", type=str, required=True)
    args = parser.parse_args()
    main(args)
