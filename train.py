# Much of the code below was imported from a notebook, and was not meant to be formatted as a script.

import os, sys, gc
from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
from tqdm.notebook import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.model_selection import KFold

from torch.cuda.amp import GradScaler, autocast
import fastai
from fastai.vision.all import *


tqdm.pandas()


@dataclass
class CFG:
    batch_size: int = 128
    num_workers: int = 2
    seed: int = 2023
    num_folds = 4
    exp_mode = True
    n_exp_epochs = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    track = True
    bio_setup = False
    exp_name = "320d_transformer"


def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict):
            yield o[item]
            continue
        elif isinstance(item, str):
            yield item
            continue
        try:
            yield from flatten(item)
        except TypeError:
            yield item


@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def before_fit(self):
        self.autocast, self.learn.scaler, self.scales = (
            autocast(),
            GradScaler(**self.kwargs),
            L(),
        )

    def before_batch(self):
        self.autocast.__enter__()

    def after_pred(self):
        if next(flatten(self.pred)).dtype == torch.float16:
            self.learn.pred = to_float(self.pred)

    def after_loss(self):
        self.autocast.__exit__(None, None, None)

    def before_backward(self):
        self.learn.loss_grad = self.scaler.scale(self.loss_grad)

    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow."
        self.skipped = True
        self.scaler.step(self)
        if self.skipped:
            raise CancelStepException()
        self.scales.append(self.scaler.get_scale())

    def after_step(self):
        self.learn.scaler.update()

    @property
    def param_groups(self):
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups

    def step(self, *args, **kwargs):
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped = False

    def after_fit(self):
        self.autocast, self.learn.scaler, self.scales = None, None, None


if __name__ == "__main__":
    iskaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    if iskaggle:
        comp_path = Path("/kaggle/input/stanford-ribonanza-rna-folding")
        data_path = Path("/kaggle/input/stanford-ribonanza-rna-folding-converted")
    else:
        comp_path = Path("../data")
        data_path = Path("../data")

    os.makedirs("./", exist_ok=True)
    df = pd.read_parquet(
        "/kaggle/input/stanford-ribonanza-rna-folding-prepare/train_data_struct_ext.parquet"
    )

    fastai.callback.fp16.MixedPrecision = MixedPrecision

    for fold in [0]:  # running multiple folds at kaggle may cause OOM
        ds_train = RNADataset(df, mode="train", fold=fold, nfolds=CFG.num_folds)
        ds_train_len = RNADataset(
            df, mode="train", fold=fold, nfolds=CFG.num_folds, mask_only=True
        )
        sampler_train = torch.utils.data.RandomSampler(ds_train_len)
        len_sampler_train = LenMatchBatchSampler(
            sampler_train, batch_size=CFG.batch_size, drop_last=True
        )
        dl_train = DeviceDataLoader(
            torch.utils.data.DataLoader(
                ds_train,
                batch_sampler=len_sampler_train,
                num_workers=CFG.num_workers,
                persistent_workers=True,
            ),
            CFG.device,
        )

        ds_val = RNADataset(df, mode="eval", fold=fold, nfolds=CFG.num_folds)
        ds_val_len = RNADataset(
            df, mode="eval", fold=fold, nfolds=CFG.num_folds, mask_only=True
        )
        sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
        len_sampler_val = LenMatchBatchSampler(
            sampler_val, batch_size=CFG.batch_size, drop_last=False
        )
        dl_val = DeviceDataLoader(
            torch.utils.data.DataLoader(
                ds_val, batch_sampler=len_sampler_val, num_workers=CFG.num_workers
            ),
            CFG.device,
        )
        gc.collect()

        data = DataLoaders(dl_train, dl_val)
        model = RNA_Model()
        model = model.to(CFG.device)
        learn = Learner(
            data,
            model,
            loss_func=loss,
            cbs=[
                GradientClip(3.0),
                SaveModelCallback(
                    monitor="mae", comp=np.less, fname=CFG.exp_name, at_end=True
                ),
            ],
            metrics=[MAE()],
        ).to_fp16()

        learn.fit_one_cycle(20, lr_max=1e-3, wd=0.05, pct_start=0.02)
        gc.collect()
