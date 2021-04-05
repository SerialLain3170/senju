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


def patch_convert(t: torch.Tensor,
                  n_crop: int,
                  min_size=1 / 8,
                  max_size=1 / 4) -> torch.Tensor:
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
    batch, ch, h, w = t.size()
    target_h = int(h * max_size)
    target_w = int(w * max_size)
    crop_h = (crop_size * h).type(torch.int64).tolist()
    crop_w = (crop_size * w).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, h - c_h)
        c_x = random.randrange(0, w - c_w)

        cropped = t[:, :, c_y: c_y + c_h, c_x: c_x + c_w]
        cropped = F.interpolate(
            cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, ch, target_h, target_w)

    return patches
