from typing import Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms

from src.datamodules.components.cxr import CXR


class CXRDataModule(LightningDataModule):
    """CXR Data Module

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,

        classes: int = 2,
        dataset_size: int = 500,
        normal: int = 1,
        img_size: int = 128,
        model: str = "Densenet121",
        scale: float = 0, 
        shear: float = 0, 
        translation: int = 0,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        rotation: int = 0,

    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.classes = classes
        self.dataset_size = dataset_size
        self.normal = normal
        self.img_size = img_size
        self.model = model
        self.scale = scale
        self.shear = shear
        self.translation = translation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation

        # data transformations

        self.transformations={"scale": self.scale, 
                            "shear": self.shear, 
                            "translation": self.translation,
                            "horizontal_flip": self.horizontal_flip,
                            "vertical_flip": self.vertical_flip,
                            "rotation": self.rotation
        }

        train_df, valid_df, test_df = self.preparing_data(path=self.data_dir)

        self.data_train: Optional[Dataset] = CXR(df = train_df, 
                                                data_dir = self.data_dir, 
                                                classes = self.classes, 
                                                model = self.model,
                                                mode='train', 
                                                transformations = self.transformations)
        self.data_val: Optional[Dataset] = CXR(df = valid_df,
                                                data_dir = self.data_dir,
                                                classes = self.classes, 
                                                model = self.model,
                                                mode='valid',
                                                normal = self.normal,
                                                img_size = self.img_size,
                                                transformations = self.transformations)
        self.data_test: Optional[Dataset] = CXR(df = test_df, 
                                                data_dir = self.data_dir, 
                                                classes = self.classes, 
                                                model = self.model, 
                                                mode='test', 
                                                normal = self.normal, 
                                                img_size = self.img_size,
                                                transformations = self.transformations)

    @property
    def num_classes(self) -> int:
        return self.classes

    def preparing_data(self, path:str):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        df_control_labels = pd.DataFrame({'id': 
            [str(i) for i in list(Path(self.data_dir, 'control').iterdir())],
            })
        df_control_labels['label'] = 0
        df_case_labels = pd.DataFrame({'id': 
            [str(i) for i in list(Path(self.data_dir, 'case').iterdir())],
            })
        df_case_labels['label'] = 1
        df = pd.concat([df_control_labels, df_case_labels]).reset_index(drop=True)
        train_df, test_df = train_test_split(df, 
                                            test_size=self.train_val_test_split[2], 
                                            random_state=42,
                                            stratify=df.label.values
                                            )
        train_df, valid_df = train_test_split(train_df,
                                            test_size=self.train_val_test_split[1]/self.train_val_test_split[0], 
                                            random_state=42,
                                            stratify=train_df.label.values
                                            )
        return train_df, valid_df, test_df

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )