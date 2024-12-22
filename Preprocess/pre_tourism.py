import torch
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import pickle as pkl
from gluonts.dataset.common import ListDataset
import csv
import warnings
from Preprocess.general import *
import argparse

warnings.filterwarnings(action="ignore", category=UserWarning, module='json')
PATH = os.path.abspath(os.path.join(os.getcwd(), "../.."))


class TourismDataset():
    """
    Tourism Dataset
    """

    def __init__(self):
        self.dataset_path = os.path.join(PATH, 'tourism')

        self.frequency_map = {
            'yearly': '12M',
            'quarterly': '3M',
            'monthly': 'M'
        }

        self.horizon_map = {
            'yearly': 4,
            'quarterly': 8,
            'monthly': 24
        }

    def load(self, frequency: str, subset_filter: str, training: bool) -> ListDataset:
        """
        Load tourism dataset.

        :param frequency:
        :param subset_filter:
        :param training:
        :return:
        """

        ds = pd.read_csv(os.path.join(f'tourism_{subset_filter}', f'{subset_filter}_in.csv'),
                         header=0, delimiter=",").T

        def extract_date(row):
            year = int(row[1])
            month = 1
            day = 1
            if subset_filter == 'quarterly':
                month = int(row[2] * 3 - 2)
            elif subset_filter == 'monthly':
                month = int(row[2])

            if month < 1 or month > 12:
                month = 1
            if year < 1:
                year = 1
            return pd.to_datetime(f'{year}-{month}-{day}')

        ds_dict = []
        meta_columns = 2 if subset_filter == 'yearly' else 3
        for item_id, row in ds.iterrows():
            ds_dict.append({
                'item_id': item_id,
                'start': extract_date(row),
                'horizon': self.horizon_map[subset_filter],
                'target': row.values[meta_columns:meta_columns + int(row[0])]
            })

        if not training:

            ds_o = pd.read_csv(os.path.join(f'tourism_{subset_filter}', f'{subset_filter}_oos.csv'),
                               header=0, delimiter=",").T
            i = 0
            for item_id, row in ds_o.iterrows():
                assert (ds_dict[i]['item_id'] == item_id)
                ds_dict[i]['target'] = np.concatenate([ds_dict[i]['target'],
                                                       row.values[meta_columns:meta_columns + int(row[0])]])
                i += 1

        return ListDataset(ds_dict, freq=self.frequency_map[subset_filter])

    def evaluate_subset(self, forecast: np.ndarray, subset_filter: str) -> np.ndarray:
        target_dataset = self.load(frequency=self.frequency_map[subset_filter],
                                   subset_filter=subset_filter,
                                   training=False)
        target = np.array([x['target'][-x['horizon']:] for x in target_dataset])
        return 100 * np.abs(forecast - target) / target

    def evaluate(self, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evaluate tourism forecasts.

        :param forecast:
        :param subset_filter:
        :return:
        """
        score_sum = 0
        time_points = 0
        scores = [
            self.evaluate_subset(forecast[:518, :4], 'tourism_yearly_m3c_quarterly'),
            self.evaluate_subset(forecast[518:945, :8], 'quarterly'),
            self.evaluate_subset(forecast[945:1311, :24], 'monthly')
        ]
        for scores_group in scores:
            for ts_scores in scores_group:
                score_sum += np.sum(ts_scores)
                time_points += len(ts_scores)
        return {
            'tourism_yearly_m3c_quarterly': float(np.mean(scores[0])),
            'quarterly': float(np.mean(scores[1])),
            'monthly': float(np.mean(scores[2])),
            'average': float(np.mean(np.array(score_sum / time_points)))
        }

    def scale(self, dataset: ListDataset) -> ListDataset:
        entries = []
        freq = None
        for entry in dataset:
            if freq is None:
                freq = entry['start'].freq
            values = entry['target']
            scaling_factor = np.abs(np.max(values))
            entries.append({**entry, 'target': values / scaling_factor, 'scaling_factor': scaling_factor})
        return ListDataset(entries, freq=freq)

    def unscale(self, dataset: ListDataset) -> np.ndarray:
        scaling_factors = np.array([r['scaling_factor'] for r in dataset])
        return scaling_factors


def pre_process_tourism(LOAD_Tourism_DATASET=False, frequency='12M', subset='yearly',
                      fpath_true_train='',
                      fpath_true_test='',
                      fpath_max_scale_train='',
                      fpath_max_scale_test='',
                      fpath_train_scaling='',
                      fpath_test_scaling=''
                      ):

    if LOAD_Tourism_DATASET:
        tourism_dataset = TourismDataset()
        train_data_tourism = tourism_dataset.load(frequency, subset, True)

        scale_train_data_tourism = tourism_dataset.scale(train_data_tourism)
        test_data_tourism = tourism_dataset.load(frequency, subset, False)
        scale_test_data_tourism = tourism_dataset.scale(test_data_tourism)

        true_train, true_test = [], []
        scale_train, scale_test = [], []
        scaling_factor_train, scaling_factor_test = [], []

        for ii in train_data_tourism:
            true_train.append(ii['target'])

        for ii in test_data_tourism:
            true_test.append(ii['target'])

        for ii in scale_train_data_tourism:
            scale_train.append(ii['target'])
            scaling_factor_train.append(ii['scaling_factor'])

        for ii in scale_test_data_tourism:
            scale_test.append(ii['target'])
            scaling_factor_test.append(ii['scaling_factor'])

        true_train = np.array(true_train, dtype=object)
        true_test = np.array(true_test, dtype=object)
        scale_train = np.array(scale_train, dtype=object)
        scale_test = np.array(scale_test, dtype=object)

        print('scale_train:', scale_train.shape, scale_test.shape)

        np.save(fpath_true_train + '.npy', true_train)
        np.save(fpath_true_test + '.npy', true_test)
        np.save(fpath_max_scale_train + '.npy', scale_train)
        np.save(fpath_max_scale_test + '.npy', scale_test)
        np.save(fpath_train_scaling + '.npy', scaling_factor_train)
        np.save(fpath_test_scaling + '.npy', scaling_factor_test)

    else:
        true_train = np.load(fpath_true_train + '.npy', allow_pickle=True)
        true_test = np.load(fpath_true_test + '.npy', allow_pickle=True)
        scale_train = np.load(fpath_max_scale_train + '.npy', allow_pickle=True)
        scale_test = np.load(fpath_max_scale_test + '.npy', allow_pickle=True)
        scaling_factor_train = np.load(fpath_train_scaling + '.npy', allow_pickle=True)
        scaling_factor_test = np.load(fpath_test_scaling + '.npy', allow_pickle=True)

        print('true_train:', true_train.shape, 'true_test:', true_test.shape,
              'scale_test:', scale_test.shape, 'scale_train:', scale_train.shape)
    return true_train, true_test, scale_train, scale_test, scaling_factor_train, scaling_factor_test


if __name__ == '__main__':
    # ===-----------------------------------------------------------------------===
    # Argument parsing
    # ===-----------------------------------------------------------------------===
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", default="M", type=str, dest="frequency", help="frequency")

    parser.add_argument("--Load_tourism", default="True", type=bool, dest="Load_m3", help="if read original data")
    parser.add_argument("--subset", default="monthly", type=str, dest="subset",
                        help="yearly | monthly | quarterly | others")
    parser.add_argument("--fpath_true_train", default="", type=str, dest="fpath_true_train",
                        help="file path for true train data")
    parser.add_argument("--fpath_true_test", default="", type=str, dest="fpath_true_test",
                        help="file path for true test data")
    parser.add_argument("--fpath_max_scale_train", default="", type=str, dest="fpath_max_scale_train",
                        help="file path for max-min-scale train data")
    parser.add_argument("--fpath_max_scale_test", default="", type=str, dest="fpath_max_scale_test",
                        help="file path for max-min-scale test data")
    parser.add_argument("--fpath_train_scaling", default="", type=str, dest="fpath_train_scaling",
                        help="file path for scaling")
    parser.add_argument("--fpath_test_scaling", default="", type=str, dest="fpath_test_scaling",
                        help="file path for test scaling")

    parser.add_argument("--ratio", default=0.01, dest="ratio", type=float,
                        help="file path for test scaling")
    parser.add_argument("--train_index_known", default="True", type=bool, dest="train_index_known",
                        help="if data index known")

    parser.add_argument("--fpath_target_train", default="", type=str, dest="fpath_target_train",
                        help="file path for selected target train")
    parser.add_argument("--fpath_target_mask", default="", type=str, dest="fpath_target_mask",
                        help="file path for selected target mask")
    parser.add_argument("--fpath_target_train_scale", default="", type=str, dest="fpath_target_train_scale",
                        help="file path for scale selected target train")
    parser.add_argument("--fpath_target_mask_scale", default="", type=str, dest="fpath_target_mask_scale",
                        help="file path for scale selected target mask")

    parser.add_argument("--fpath_split_target_train", default="", type=str, dest="fpath_split_target_train",
                        help="file path for split target train")
    parser.add_argument("--fpath_split_target_mask", default="", type=str, dest="fpath_split_target_mask",
                        help="file path for split target mask")

    parser.add_argument("--fpath_split_scale_target_train", default="", type=str, dest="fpath_split_scale_target_train",
                        help="file path for split scale target train")
    parser.add_argument("--fpath_split_target_mask", default="", type=str, dest="fpath_split_scale_target_mask",
                        help="file path for split scale target mask")
    parser.add_argument("--seq_len", default=12, type=int, dest="seq_len",
                        help="sequence length")

    options = parser.parse_args()
    pre_process_tourism(LOAD_Tourism_DATASET=options.Load_tourism,
                   frequency=options.frequency, subset=options.sub_set,
                   fpath_true_train=options.fpath_true_train, fpath_true_test=options.fpath_true_test,
                   fpath_max_scale_train=options.fpath_max_scale_train,
                   fpath_max_scale_test=options.fpath_max_scale_test,
                   fpath_train_scaling=options.fpath_train_scaling,
                   fpath_test_scaling=options.fpath_test_scaling
                   )

    get_dataset(fpath_true_train=options.fpath_true_train, fpath_true_test=options.fpath_true_test,
                fapth_scale_train=options.fapth_scale_train, fpath_target_train=options.fpath_target_train,
                fpath_target_mask=options.fpath_target_mask, fpath_target_train_scale=options.fpath_target_train_scale,
                fpath_target_mask_scale=options.fpath_target_mask_scale, fpath_train_index=options.fpath_train_index,
                fpath_mask_index=options.fpath_mask_index, ratio=options.ratio,
                train_index_known=options.train_index_known)

    get_split_dataset(fpath_target_train=options.fpath_target_train, fpath_target_mask=options.fpath_target_mask,
                      fpath_target_train_scale=options.fpath_target_train_scale,
                      fpath_target_mask_scale=options.fpath_target_mask_scale,

                      fpath_target_train_save=options.fpath_split_target_train,
                      fpath_target_mask_save=options.fpath_split_target_mask,
                      fpath_scale_target_train_save=options.fpath_split_scale_target_train,
                      fpath_scale_target_mask_save=options.fpath_split_scale_target_mask,
                      seq_len=options.seq_len)

