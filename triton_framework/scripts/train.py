#!/usr/bin/env python3
from __future__ import annotations
import os
import yaml
import torch
from torch.utils.data import DataLoader

from triton_framework.engine.utils import set_seed, get_device, OptimConfig
from triton_framework.engine.trainer import Trainer, TrainConfig
from triton_framework.data.dummy import DummyLM
from triton_framework.modules.simple_lm import TinyLM


def main():
    cfg_path = os.environ.get("TFW_CONFIG", os.path.join(os.path.dirname(__file__), "../src/triton_framework/configs/default.yaml"))
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("train", {}).get("device", "auto"))

    d_model = cfg["model"]["d_model"]
    hidden = cfg["model"]["hidden"]
    rms_eps = cfg["model"].get("rms_eps", 1e-5)
    use_triton = cfg["model"].get("use_triton", True)

    ds = DummyLM(cfg["data"]["num_samples"], cfg["data"]["seq_len"], cfg["data"]["vocab_size"]) 
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    model = TinyLM(d_model, hidden, cfg["data"]["vocab_size"], rms_eps=rms_eps, use_triton=use_triton)

    trainer = Trainer(
        model,
        dl,
        device,
        optim_cfg=OptimConfig(lr=cfg["train"]["lr"]),
        cfg=TrainConfig(
            max_steps=cfg["train"]["max_steps"],
            log_interval=cfg["train"]["log_interval"],
            amp=cfg["train"].get("amp", True),
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()
