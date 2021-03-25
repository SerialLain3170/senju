import torch
import random
import shutil
import datetime
import torch.nn.functional as F

from pathlib import Path


def session(session_name):
    session_path = Path("session") / Path(session_name)
    if session_path.exists():
        dt = datetime.datetime.now()
        dt = dt.strftime('%m%d-%H%M-%S%f')[:-4]
        session_name = f"{session_name}.{dt}"
        session_path = Path("session") / Path(session_name)

    modeldir_path = session_path / "ckpts"
    outdir_path = session_path / "vis"

    modeldir_path.mkdir(exist_ok=True, parents=True)
    outdir_path.mkdir(exist_ok=True, parents=True)

    shutil.copy("param.yaml", session_path)

    return outdir_path, modeldir_path


def call_zeros(tensor: torch.Tensor):
    zeros = torch.zeros_like(tensor).cuda()

    return zeros


def call_ones(tensor: torch.Tensor):
    ones = torch.ones_like(tensor).cuda()

    return ones
