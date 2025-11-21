#!/bin/bash

# paths
export SCRATCH=/scratch/$USER
export PERMANENT=/scratch/work/yang-lab/users/$USER

# conda
export CONDA_ENV=$PERMANENT/conda_envs/reb

# singularity
export SINGULARITY_IMG=/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif
export SINGULARITY_OVERLAY=$PERMANENT/overlays/wm-repr/overlay-15GB-500K.ext3
export SINGULARITYENV_TFDS_DATA_DIR=$SCRATCH/tfds
export SINGULARITYENV_XDG_CACHE_HOME=$SCRATCH/.cache
export SINGULARITYENV_HF_HOME=$SCRATCH/.cache/huggingface
export SINGULARITYENV_TORCH_HOME=$SCRATCH/.cache/torch
# necessary to avoid: wandb.errors.errors.CommError: Failed to get resume status for run nxsbcnvp: api: failed sending: POST https://api.wandb.ai/graphql giving up after 1 attempt(s): Post "https://api.wandb.ai/graphql": tls: failed to verify certificate: x509: certificate signed by unknown authority
export SINGULARITYENV_SSL_CERT_FILE=/home/ms16518/cacert.pem

# wandb
export WANDB_ENTITY=mattia-scardecchia
export WANDB_PROJECT=rebuttal
export WANDB_API_KEY=aaaa6a0838d0e4f2d3f68ffbc76dcb40602660cf

# stack tracebacks
export HYDRA_FULL_ERROR=1