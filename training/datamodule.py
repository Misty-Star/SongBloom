"""SongBloom Lightning DataModule."""

import lightning as pl
from torch.utils.data import DataLoader

from .dataset import SongBloomDataset, collate_fn


class SongBloomDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_dir: str,
        val_dir: str = "",
        sample_rate: int = 48000,
        block_size: int = 16,
        max_duration: float = 240.0,
        prompt_len: float = 10.0,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        common = dict(
            sample_rate=self.hparams.sample_rate,
            block_size=self.hparams.block_size,
            max_duration=self.hparams.max_duration,
            prompt_len=self.hparams.prompt_len,
        )
        self.train_ds = SongBloomDataset(self.hparams.train_dir, **common)
        if self.hparams.val_dir:
            self.val_ds = SongBloomDataset(self.hparams.val_dir, **common)
        else:
            self.val_ds = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
