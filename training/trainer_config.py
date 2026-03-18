"""Trainer 参数组装辅助函数。"""

import typing as tp


def get_trainer_root_dir() -> str:
    return "training/runs"


def get_validation_trainer_kwargs(
    val_dir: tp.Optional[str],
    val_check_interval: int,
) -> dict:
    if not val_dir or not str(val_dir).strip():
        return {
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
        }

    return {
        "val_check_interval": val_check_interval,
    }
