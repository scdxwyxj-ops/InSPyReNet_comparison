import re
import os
import json
import yaml
import cv2
import argparse
import warnings
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as ed

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "CONSTANT.json"

class Simplify(nn.Module):
    def __init__(self, model):
        super(Simplify, self).__init__()
        self.model = model
        
    def cuda(self):
        self.model = self.model.cuda()
        return self
        
    def forward(self, x):
        out = self.model({'image': x})
        return out['pred']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  '-c', type=str,            default='configs/InSPyReNet_SwinB.yaml')
    parser.add_argument('--resume',  '-r', action='store_true', default=False)
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    parser.add_argument('--debug',   '-d', action='store_true', default=False)
    args = parser.parse_args()
    
    cuda_visible_devices = None
    local_rank = -1

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        cuda_visible_devices = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
    if "LOCAL_RANK" in os.environ.keys():
        local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == -1:
        device_num = 1
    elif cuda_visible_devices is None:
        device_num = torch.cuda.device_count()
    else:
        device_num = len(cuda_visible_devices)

    args.device_num = device_num
    args.local_rank = local_rank
    
    warnings.simplefilter("ignore")

    return args

def sort(x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(x, key=alphanum_key)

def load_config(config_dir, easy=True):
    cfg = yaml.load(open(config_dir), yaml.FullLoader)
    if easy is True:
        cfg = ed(cfg)
    _apply_constant_data_root(cfg)
    return cfg

def _apply_constant_data_root(cfg):
    data_root = _load_constant_data_root()
    if data_root is None:
        return

    repo_root = Path(__file__).resolve().parents[1]

    if hasattr(cfg, "Train") and hasattr(cfg.Train, "Dataset"):
        cfg.Train.Dataset.root = _resolve_root(cfg.Train.Dataset.root, data_root, repo_root)
    if hasattr(cfg, "Test") and hasattr(cfg.Test, "Dataset"):
        cfg.Test.Dataset.root = _resolve_root(cfg.Test.Dataset.root, data_root, repo_root)
    if hasattr(cfg, "Eval"):
        cfg.Eval.gt_root = _resolve_root(getattr(cfg.Eval, "gt_root", None), data_root, repo_root)

def _load_constant_data_root():
    if not _CONFIG_PATH.exists():
        return None

    config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_path = config.get("data_path")
    if not raw_path:
        return None
    return _normalise_path(raw_path)

def _resolve_root(root, data_root, repo_root):
    if root is None:
        return root

    root_path = Path(root)
    if root_path.is_absolute():
        if root_path.exists():
            return str(root_path)
    else:
        repo_candidate = (repo_root / root_path).resolve()
        if repo_candidate.exists():
            return str(repo_candidate)

    if data_root is not None:
        data_root = Path(data_root)
        data_candidate = (data_root / root_path)
        if data_candidate.exists():
            return str(data_candidate)
        if data_root.exists():
            return str(data_root)

    return root

def _normalise_path(raw_path):
    normalised = raw_path.replace("\\", "/")
    candidate = Path(normalised)
    if candidate.exists():
        return candidate

    if len(normalised) > 2 and normalised[1:3] == ":/":
        drive_letter = normalised[0].lower()
        remainder = normalised[3:]
        wsl_candidate = Path("/mnt") / drive_letter / remainder
        if wsl_candidate.exists():
            return wsl_candidate
        return wsl_candidate

    return candidate

def to_cuda(sample):
    for key in sample.keys():
        if type(sample[key]) == torch.Tensor:
            sample[key] = sample[key].cuda()
    return sample

def to_numpy(pred, shape):
    pred = F.interpolate(pred, shape, mode='bilinear', align_corners=True)
    pred = pred.data.cpu()
    pred = pred.numpy().squeeze()
    return pred

def debug_tile(deblist, size=(100, 100), activation=None):
    debugs = []
    for debs in deblist:
        debug = []
        for deb in debs:
            if activation is not None:
                deb = activation(deb)
            log = deb.cpu().detach().numpy().squeeze()
            log = ((log - log.min()) / (log.max() - log.min()) * 255).astype(np.uint8)
            log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
            log = cv2.resize(log, size)
            debug.append(log)
        debugs.append(np.vstack(debug))
    return np.hstack(debugs)


if __name__ == "__main__":
    x = torch.rand(4, 3, 576, 576)
