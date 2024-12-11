# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of this file is based on
# https://github.com/karpathy/nanoGPT/blob/master/train.py

"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import inspect
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.flop_counter import FlopCounterMode
from triton.testing import do_bench

import wandb
from diffusion import create_diffusion
from models import DiT_models


def get_flops_achieved(f):
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        f()
    total_flops = flop_counter.get_total_flops()
    ms_per_iter = do_bench(f)
    iters_per_second = 1e3 / ms_per_iter
    return f"{iters_per_second * total_flops / 1e12: .2f} TF/s"


def configure_optimizers(
    model, weight_decay, learning_rate, betas, device_type, logger
):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate},
        {"params": nodecay_params, "weight_decay": 0.0, "lr": learning_rate},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"using fused AdamW: {use_fused}")

    return optimizer


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if RANK == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


# CUDA setup
assert torch.cuda.is_available()

# DDP setup
RANK = 0
DEVICE = 0
WORLD_SIZE = 1
IS_DISTRIBUTED = False
if "LOCAL_RANK" in os.environ:  # torchrun setup
    dist.init_process_group("nccl")
    RANK = dist.get_rank()
    DEVICE = RANK % torch.cuda.device_count()
    WORLD_SIZE = dist.get_world_size()
    IS_DISTRIBUTED = True


# Parse config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
with open(args.config, "r") as f:
    CONFIG = yaml.safe_load(f)
opt_config = CONFIG["optimization"]

# Validate batch size and set seed
assert opt_config["global_batch_size"] % WORLD_SIZE == 0
seed = CONFIG["seed"] * WORLD_SIZE + RANK
torch.manual_seed(seed)
torch.cuda.set_device(DEVICE)

# Initialize wandb if enabled
if RANK == 0 and CONFIG["enable_wandb"]:
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)

# Setup experiment directories and logger
if RANK == 0:
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    experiment_index = len(glob(f"{CONFIG['results_dir']}/*"))
    model_string_name = CONFIG["model"].replace("/", "-")
    experiment_dir = (
        f"{CONFIG['results_dir']}/{experiment_index:03d}-{model_string_name}"
    )
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
else:
    logger = create_logger(None)

# Log config and setup info
logger.info("Config:")
for key, value in CONFIG.items():
    logger.info(f"  {key}: {value}")
logger.info(f"Starting rank={RANK}, seed={seed}, world_size={WORLD_SIZE}.")


def get_batch(step, batch_size, seq_len):
    import numpy as np

    # Load dataset from memmap file

    arr = np.memmap(CONFIG["data_path"], dtype=np.float16, mode="r")
    arr = np.memmap(
        CONFIG["data_path"],
        dtype=np.float16,
        mode="r",
        shape=(arr.shape[0] // 102, 102),
    )

    # Create random number generator
    seed = step * WORLD_SIZE + RANK
    rng = np.random.Generator(np.random.PCG64(seed=seed))

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


# Initialize model
model = DiT_models[CONFIG["model"]](
    input_size=CONFIG["input_size"],
    in_channels=CONFIG["in_channels"],
    learn_sigma=CONFIG["learn_sigma"],
).float()

# Setup model and EMA
ema = deepcopy(model).to(DEVICE)
requires_grad(ema, False)
model = model.to(DEVICE)
simple_model = model


if IS_DISTRIBUTED:
    model = DDP(model, device_ids=[RANK])

if CONFIG["enable_compile"]:
    model = torch.compile(model, dynamic=True)


update_ema(ema, simple_model, decay=0)

# Setup diffusion
diffusion = create_diffusion(timestep_respacing="", learn_sigma=CONFIG["learn_sigma"])
logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Configure optimizer
opt = configure_optimizers(
    model=model,
    weight_decay=opt_config["weight_decay"],
    learning_rate=opt_config["learning_rate"],
    betas=(opt_config["betas"]["beta1"], opt_config["betas"]["beta2"]),
    device_type="cuda",
    logger=logger,
)

# Load checkpoint if resuming training
train_steps = 0
if CONFIG["resume_from_ckpt"]:
    if os.path.exists(CONFIG["resume_from_ckpt"]):
        logger.info(f"Loading checkpoint from {CONFIG['resume_from_ckpt']}")
        checkpoint = torch.load(
            CONFIG["resume_from_ckpt"],
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        checkpoint["model"]["pos_embed"] = torch.clone(simple_model.pos_embed)
        checkpoint["ema"]["pos_embed"] = torch.clone(simple_model.pos_embed)
        simple_model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        train_steps = checkpoint["train_steps"]
        logger.info(f"Resuming from step {train_steps}")
        torch.manual_seed(seed + train_steps)
    else:
        raise FileNotFoundError(
            f"Checkpoint file {CONFIG['resume_from_ckpt']} not found."
        )


# Prepare models for training
model.train()
ema.eval()

# Initialize training monitoring variables
log_steps = 0
running_loss = 0
start_time = time()


def compute_loss(model, x, phone, speaker_id, length):
    """
    Compute loss for a batch of data
    """
    x = x[..., :length].to(DEVICE).float()
    phone = phone[..., :length].to(DEVICE)
    speaker_id = speaker_id.to(DEVICE)
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
    model_kwargs = dict(phone=phone, speaker_id=speaker_id)
    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    loss = loss_dict["loss"].float().mean()
    return loss


while True:
    if train_steps % CONFIG["ckpt_every"] == 0:
        seed = CONFIG["seed"] * (train_steps + 1) * WORLD_SIZE + RANK
        torch.manual_seed(seed)

    length = int(2 ** (5 + train_steps / 10_000))
    length = min(length, CONFIG["input_size"])
    batch_size = (
        opt_config["global_batch_size"]
        // WORLD_SIZE
        // int(2 ** int(train_steps / 10_000))
    )

    x, (speaker_id, phone) = get_batch(train_steps, batch_size, seq_len=length)

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=CONFIG["use_bfloat16"]
    ):
        loss = compute_loss(model, x, phone, speaker_id, length=length)
    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), opt_config["max_grad_norm"]
    )
    opt.step()
    update_ema(ema, simple_model)

    running_loss += loss
    log_steps += 1
    train_steps += 1

    # Save DiT checkpoint:
    if train_steps % CONFIG["ckpt_every"] == 0 and train_steps > 0:
        if RANK == 0:
            checkpoint = {
                "model": simple_model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "config": CONFIG,
                "train_steps": train_steps,
                "batch_size": batch_size,
            }
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        if IS_DISTRIBUTED:
            dist.barrier()

    if train_steps % CONFIG["log_every"] == 0:
        # Measure training speed:
        torch.cuda.synchronize()
        end_time = time()
        steps_per_sec = log_steps / (end_time - start_time)
        # Reduce loss history over all processes:
        avg_loss = torch.tensor(running_loss.item() / log_steps, device=DEVICE)
        if IS_DISTRIBUTED:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / WORLD_SIZE
        logger.info(
            f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Grad Norm: {grad_norm:.4f} Length: {length}"
        )

        # Log to wandb
        if RANK == 0 and CONFIG["enable_wandb"]:
            wandb.log(
                {
                    "avg_loss": avg_loss,
                    "steps_per_sec": steps_per_sec,
                    "step": train_steps,
                    "grad_norm": grad_norm,
                    "length": length,
                }
            )

        # Reset monitoring variables:
        running_loss = 0
        log_steps = 0
        start_time = time()


logger.info("Done!")

if RANK == 0:
    wandb.finish()

if IS_DISTRIBUTED:
    dist.destroy_process_group()
