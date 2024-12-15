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
import math
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import torch
from torch.nn.attention.flex_attention import create_block_mask

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
        f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"Use fused AdamW: {use_fused}")

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

model_config = CONFIG["model"]
opt_config = CONFIG["optimization"]
data_config = CONFIG["data"]
training_config = CONFIG["training"]
wandb_config = training_config["wandb"]

# Validate batch size and set seed
assert opt_config["global_batch_size"] % WORLD_SIZE == 0
seed = training_config["seed"] * WORLD_SIZE + RANK
torch.manual_seed(seed)
torch.cuda.set_device(DEVICE)

# Initialize wandb if enabled
if RANK == 0 and wandb_config["enable"]:
    wandb.init(project=wandb_config["project"], config=CONFIG)

# Setup experiment directories and logger
if RANK == 0:
    os.makedirs(training_config["results_dir"], exist_ok=True)
    experiment_index = len(glob(f"{training_config['results_dir']}/*"))
    model_string_name = model_config["name"].replace("/", "-")
    experiment_dir = (
        f"{training_config['results_dir']}/{experiment_index:03d}-{model_string_name}"
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
    data_dim = data_config["data_dim"]
    data_path = data_config["data_path"]
    arr = np.memmap(data_path, dtype=np.float16, mode="r")
    arr = np.memmap(
        data_path,
        dtype=np.float16,
        mode="r",
        shape=(arr.shape[0] // (data_dim + 2), data_dim + 2),
    )

    # Create random number generator
    seed = step * WORLD_SIZE + RANK
    rng = np.random.Generator(np.random.PCG64(seed=seed))

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
    if data_config["normalize"]:
        x = (x - data_config["data_mean"]) / data_config["data_std"]
    phone = torch.from_numpy(phone).to(DEVICE)
    speaker_id = torch.from_numpy(speaker_id).to(DEVICE)

    return x, (speaker_id, phone)


# Initialize model
model = DiT_models[model_config["name"]](
    input_size=model_config["input_size"],
    embedding_vocab_size=model_config["embedding_vocab_size"],
    learn_sigma=model_config["learn_sigma"],
    in_channels=data_config["data_dim"],
).float()

# log model architecture
logger.info(f"Model architecture: {model}")

# Setup model and EMA
ema = deepcopy(model).to(DEVICE)
requires_grad(ema, False)
model = model.to(DEVICE)
simple_model = model


if IS_DISTRIBUTED:
    model = DDP(model, device_ids=[RANK])

use_compile = training_config.get("enable_compile", False)
if use_compile:
    model = torch.compile(model, dynamic=True)
logger.info(f"Use torch.compile: {use_compile}")


update_ema(ema, simple_model, decay=0)

# Setup diffusion
learn_sigma = model_config["learn_sigma"]
diffusion = create_diffusion(timestep_respacing="", learn_sigma=learn_sigma)
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
ckpt_path = training_config.get("resume_from_ckpt", None)
if ckpt_path:
    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(
            ckpt_path, map_location=lambda storage, loc: storage, weights_only=False
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
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")


# Prepare models for training
model.train()
ema.eval()

# Initialize training monitoring variables
log_steps = 0
running_loss = 0
start_time = time()


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < opt_config["warmup_iters"]:
        return opt_config["learning_rate"] * (it + 1) / (opt_config["warmup_iters"] + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > opt_config["lr_decay_iters"]:
        return opt_config["min_lr"]
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - opt_config["warmup_iters"]) / (
        opt_config["lr_decay_iters"] - opt_config["warmup_iters"]
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return opt_config["min_lr"] + coeff * (
        opt_config["learning_rate"] - opt_config["min_lr"]
    )


def compute_loss(model, x, phone, speaker_id, length):
    """
    Compute loss for a batch of data
    """
    x = x[..., :length].to(DEVICE).float()
    phone = phone[..., :length].to(DEVICE)
    speaker_id = speaker_id.to(DEVICE)
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
    if training_config.get("use_block_mask", False):
        B = speaker_id.shape[0]
        S = speaker_id.shape[1]

        def document_masking(b, h, q_idx, kv_idx):
            del h
            A = speaker_id[b, q_idx]
            B = speaker_id[b, kv_idx]
            return A == B

        attn_mask = create_block_mask(
            document_masking,
            B=B,
            H=1,
            Q_LEN=S,
            KV_LEN=S,
            device="cuda",
            _compile=True,
            BLOCK_SIZE=128,
        )
    else:
        attn_mask = speaker_id[:, None, :] == speaker_id[:, :, None]
        attn_mask = attn_mask.unsqueeze(1)
    model_kwargs = dict(phone=phone, speaker_id=speaker_id, attn_mask=attn_mask)
    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    loss = loss_dict["loss"].float().mean()
    return loss


use_bfloat16 = training_config.get("use_bfloat16", False)
logger.info(f"Use bfloat16: {use_bfloat16}")

logger.info(f"Use learning rate decay: {opt_config['decay_lr']}")

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(train_steps) if opt_config["decay_lr"] else opt_config["learning_rate"]
    for param_group in opt.param_groups:
        param_group["lr"] = lr

    if train_steps % training_config["ckpt_every"] == 0:
        seed = training_config["seed"] * (train_steps + 1) * WORLD_SIZE + RANK
        logger.info(f"Setting seed to {seed} at step {train_steps}")
        torch.manual_seed(seed)

    length = int(opt_config["initial_input_size"] * 2 ** (train_steps / 10_000))
    length = min(length, model_config["input_size"])
    batch_size = opt_config["global_batch_size"] // WORLD_SIZE
    if opt_config["constant_memory"]:
        batch_size = batch_size // (length // opt_config["initial_input_size"])

    x, (speaker_id, phone) = get_batch(train_steps, batch_size, seq_len=length)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
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
    if train_steps % training_config["ckpt_every"] == 0 and train_steps > 0:
        if RANK == 0:
            checkpoint = {
                "model": simple_model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "config": CONFIG,
                "train_steps": train_steps,
                "batch_size": batch_size,
                "lr": lr,
            }
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        if IS_DISTRIBUTED:
            dist.barrier()

    if train_steps % training_config["log_every"] == 0:
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
        if RANK == 0 and wandb_config["enable"]:
            wandb.log(
                {
                    "avg_loss": avg_loss,
                    "steps_per_sec": steps_per_sec,
                    "step": train_steps,
                    "grad_norm": grad_norm,
                    "length": length,
                    "lr": lr,
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
