"""
In this file resides the training loop
"""

import torch
import pandas as pd

from data_pipeline import *
from models import *
from process_laplacian import scipy2torch


def train_impainter(impainter_model: Impaint, impainter_dataset: GHCNImpainter, optimizer, date_start, date_end, num_epochs):
    # HERE WE make batch size == sampling times
    # then a batch == a day
    times_sampling = impainter_dataset.sampler.times
    for epoch in range(num_epochs):
        for date_idx, _ in enumerate(pd.date_range(date_start, date_end)):
            # further_masked_laplacian, masked_x are of the same shape
            # idx_to_mask, masked_value don't have to be the same size
            for resample_idx in range(times_sampling):
                idx = date_idx * times_sampling + resample_idx
                further_masked_laplacian, masked_x, masked_idx, masked_value = impainter_dataset[idx]
                further_masked_laplacian = scipy2torch(further_masked_laplacian)
                masked_x = masked_x.unsqueeze(dim=0)  # batch=1 according to def in layers.py
                out = impainter_model.forward(further_masked_laplacian, masked_x)  # out should be shaped 1 x V x 1
                out = out.squeeze()
                # compute loss, eventually
                pred = out[masked_idx]
                loss = torch.sum((masked_value - pred) ** 2) / len(masked_idx)
                print("EPOCH: {}; BATCH_IDX: {}; RESAMPLING_IDX: {}; LOSS: {}"
                      .format(epoch, date_idx, resample_idx, loss.item()))
                impainter_model.zero_grad()
                loss.backward()
                optimizer.step()

