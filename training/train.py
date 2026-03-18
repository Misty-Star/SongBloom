"""SongBloom 训练入口脚本。

用法:
    python -m training.train --config training/configs/songbloom_full_240s_train.yaml

分布式训练:
    python -m training.train --config training/configs/songbloom_full_240s_train.yaml --num-gpus 8
"""

import argparse
import os
import sys

from omegaconf import OmegaConf
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DeepSpeedStrategy

# 项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SongBloom.models.songbloom.songbloom_pl import SongBloom_PL
from training.datamodule import SongBloomDataModule
from training.trainer_config import (
    get_trainer_root_dir,
    get_validation_trainer_kwargs,
)


def load_config(cfg_file):
    OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx], replace=True)
    OmegaConf.register_new_resolver("get_fname", lambda x: os.path.splitext(os.path.basename(x))[0], replace=True)
    OmegaConf.register_new_resolver("load_yaml", lambda x: OmegaConf.load(x), replace=True)
    OmegaConf.register_new_resolver("dynamic_path", lambda x: x.replace("???", os.path.dirname(cfg_file)), replace=True)
    return OmegaConf.load(cfg_file)


def main():
    parser = argparse.ArgumentParser(description="SongBloom Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 构建模型
    model = SongBloom_PL(cfg)

    # 从预训练权重初始化（如果指定）
    pretrained = getattr(cfg, "pretrained_path", None)
    if pretrained and os.path.exists(pretrained):
        import torch
        state = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {pretrained}")

    # 构建数据
    data_cfg = cfg.data
    dm = SongBloomDataModule(
        train_dir=data_cfg.train_dir,
        val_dir=getattr(data_cfg, "val_dir", ""),
        sample_rate=cfg.sr,
        block_size=cfg.model.block_size,
        max_duration=cfg.max_dur,
        prompt_len=cfg.train_dataset.prompt_len,
        batch_size=getattr(data_cfg, "batch_size", 4),
        num_workers=getattr(data_cfg, "num_workers", 4),
    )

    # 回调
    callbacks = [
        ModelCheckpoint(
            every_n_train_steps=5000,
            save_top_k=3,
            monitor="train/loss",
            filename="step{step:06d}-loss{train/loss:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # 训练策略
    train_cfg = getattr(cfg, "training", OmegaConf.create())
    if args.num_gpus > 1:
        strategy = DeepSpeedStrategy(stage=2)
    else:
        strategy = "auto"

    validation_trainer_kwargs = get_validation_trainer_kwargs(
        val_dir=getattr(data_cfg, "val_dir", ""),
        val_check_interval=getattr(train_cfg, "val_check_interval", 5000),
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=strategy,
        default_root_dir=get_trainer_root_dir(),
        precision=getattr(cfg, "precision", "bf16-mixed"),
        max_steps=getattr(train_cfg, "max_steps", 150000),
        accumulate_grad_batches=getattr(train_cfg, "accumulate_grad_batches", 1),
        gradient_clip_val=getattr(train_cfg, "gradient_clip_val", 1.0),
        callbacks=callbacks,
        log_every_n_steps=10,
        **validation_trainer_kwargs,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
