import os
import sys

from itertools import chain
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import process_image

from pathlib import Path

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


NUM_THREADS = cpu_count()

MAPPING_CLASS_INV = ['He_NR', 'ER']
MAPPING_CLASS = {v: k for k, v in enumerate(MAPPING_CLASS_INV)}

MAPPING_VALUE_INV = [1, 3, 6, 10, 20, 30]
MAPPING_VALUE = {v: k for k, v in enumerate(MAPPING_VALUE_INV)}
MAPPING_VALUE_INV = np.asarray(MAPPING_VALUE_INV)


class IDAODatasetInference(Dataset):
    def __init__(self, path, transform_fn):
        super(IDAODatasetInference, self).__init__()

        self.images_paths = Path(path).rglob('*.png')
        self.images_paths = sorted(map(str, self.images_paths))

        self.transform_fn = transform_fn

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx

        if not (0 <= idx < len(self)):
            raise IndexError(idx)

        image, *_, image_name = process_image(self.images_paths[idx], check_target=False)
        if self.transform_fn is not None:
            image = self.transform_fn(image=image)
        return {'image': image, 'name': image_name}

    def __len__(self):
        return len(self.images_paths)


def collate_fn(batch):
    names = [r['name'] for r in batch]

    images = [r['image'] for r in batch]
    images = np.asarray(images)
    images = torch.tensor(images, dtype=torch.float, device='cpu').unsqueeze(1)

    return images, names


class IDAONet(nn.Module):
    def __init__(self):
        super(IDAONet, self).__init__()

        self._conv = nn.Sequential(
            nn.BatchNorm2d(1),

            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3),

            nn.Flatten(),
            nn.ReLU(),
        )

        self._cls_labels = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(6272, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 1),
        )

        self._cls_values = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(6272, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, len(MAPPING_VALUE_INV)),
        )

    def forward(self, images):
        images = self._conv(images)
        labels = self._cls_labels(images)
        values = self._cls_values(images)
        return labels, values


def make_submission(model, iterator, verbose=False):
    model.eval()

    filenames_all = []
    labels_pred_all, values_pred_all = [], []

    if verbose:
        from tqdm import tqdm
        iterator_ = tqdm(iterator)
    else:
        iterator_ = iterator

    with torch.no_grad():
        for images, *_, filenames in iterator_:
            labels_pred, values_pred = model(images)

            labels_pred = torch.sigmoid(labels_pred)
            labels_pred = labels_pred.cpu().detach().numpy().ravel()
            values_pred = values_pred.cpu().detach().numpy().argmax(axis=1).ravel()

            labels_pred_all.append(labels_pred)
            values_pred_all.append(values_pred)

            filenames_all.extend(filenames)

    labels_pred = np.concatenate(labels_pred_all)
    values_pred = np.concatenate(values_pred_all)
    values_pred = MAPPING_VALUE_INV[values_pred]

    submission = pd.DataFrame()
    submission['id'] = filenames_all
    submission['classification_predictions'] = labels_pred
    submission['regression_predictions'] = values_pred

    submission['id'] = submission['id'].str.split('.').str[0]
    submission.set_index('id', inplace=True)

    return submission


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='IDAO-2021')
    parser.add_argument('-m', '--model', nargs='?', required=True, help='model path')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

    args = parser.parse_args()

    net = IDAONet()

    model_paths = Path(args.model).rglob('*.pt')
    model_paths = sorted(map(str, model_paths))

    dataset_public = DataLoader(
        dataset=IDAODatasetInference(
            path='tests/public_test/',
            transform_fn=None,
        ),
        batch_size=128,
        shuffle=False,
        num_workers=NUM_THREADS,
        collate_fn=collate_fn,
    )

    dataset_private = DataLoader(
        dataset=IDAODatasetInference(
            path='tests/private_test/',
            transform_fn=None,
        ),
        batch_size=128,
        shuffle=False,
        num_workers=NUM_THREADS,
        collate_fn=collate_fn,
    )

    submission_pull = []
    
    for fold, model_best in enumerate(model_paths):
        net.load_state_dict(torch.load(model_best, map_location=torch.device('cpu')))

        submission_pub = make_submission(net, dataset_public, verbose=args.verbose)
        submission_prv = make_submission(net, dataset_private, verbose=args.verbose)

        submission = pd.concat([submission_pub, submission_prv], axis=0)
        submission = submission.add_suffix(f'_{fold}')
        submission_pull.append(submission)

    submission_pull = pd.concat(submission_pull, axis=1)

    submission = pd.DataFrame(index=submission_pull.index)

    mask = submission_pull.columns.str.startswith('classification_predictions_')
    submission['classification_predictions'] = submission_pull.loc[:, mask].mean(axis=1).values

    mask = submission_pull.columns.str.startswith('regression_predictions_')
    submission['regression_predictions'] = submission_pull.loc[:, mask].mode(axis=1).mean(axis=1).values.astype(int)

    submission.to_csv('submission.csv', index=True)
