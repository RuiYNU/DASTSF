import torch
import numpy as np
from typing import Dict
import pandas as pd
from gluonts.dataset.common import ListDataset
import os
import warnings
import csv
import argparse
from Preprocess.general import *
warnings.filterwarnings(action="ignore", category=UserWarning, module='json')


class M3Dataset():
    def __init__(self, fpath_dataset_file):
        # self.dataset_file = os.path.join('./', 'M3C.xls')
        self.dataset_file = fpath_dataset_file

        self.sheet_map = {
            'yearly': 'M3Year',
            'quarterly': 'M3Quart',
            'monthly': 'M3Month',
            'others': 'M3Other',
        }
        self.frequency_map = {
            'tourism_yearly_m3c_quarterly': '12M',
            'quarterly': '3M',
            'monthly': 'M',
            'others': 'H'
        }

    def load(self, frequency: str, subset_filter: str, training: bool) -> ListDataset:
        """
        Load M3 dataset.

        :param frequency: Frequency (gluonts format).
        :param subset_filter: one of [tourism_yearly_m3c_quarterly, quarterly, monthly, others]
        :param training: Whole dataset if False. Without last horizon if True.
        :return: ListDataset (gluonts format).
        """
        meta_columns = 6
        dataset = pd.read_excel(self.dataset_file, sheet_name=self.sheet_map[subset_filter])

        def extract_date(row):
            year = row['Starting Year'] if subset_filter != 'others' else 1
            month = 1
            day = 1
            if subset_filter == 'quarterly':
                month = row['Starting Quarter'] * 3 - 2
            elif subset_filter == 'monthly':
                month = row['Starting Month']
            if month < 1 or month > 12:
                month = 1
            if year < 1:
                year = 1
            return pd.to_datetime(f'{year}-{month}-{day}')

        def holdout(row) -> int:
            return row['NF'] if training else 0

        items_all = [{
            'item_id': row['Series'],
            'start': extract_date(row),
            'horizon': row['NF'],
            'target': row.values[meta_columns:meta_columns + row['N'] - holdout(row)]}
            for _, row in dataset.iterrows()]

        return ListDataset(items_all, freq=frequency)

    def evaluate_subset(self, forecast: np.ndarray, subset_filter: str) -> np.ndarray:
        target_dataset = self.load(frequency=self.frequency_map[subset_filter],
                                   subset_filter=subset_filter,
                                   training=False)

        target = np.array([x['target'][-x['horizon']:] for x in target_dataset])
        return 200 * np.abs(forecast - target) / (forecast + target)

    def evaluate(self, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecast.

        :param forecast: Forecast to evaluate.
        :return: Scores for each forecast point
        """
        score_sum = 0
        time_points = 0
        scores = [
            self.evaluate_subset(forecast[:645, :6], 'tourism_yearly_m3c_quarterly'),
            self.evaluate_subset(forecast[645:1401, :8], 'quarterly'),
            self.evaluate_subset(forecast[1401:2829, :18], 'monthly'),
            self.evaluate_subset(forecast[2829:3003, :8], 'others'),
        ]
        for scores_group in scores:
            for ts_scores in scores_group:
                score_sum += np.sum(ts_scores)
                time_points += len(ts_scores)
        return {
                   'tourism_yearly_m3c_quarterly': float(np.mean(scores[0])),
                   'quarterly': float(np.mean(scores[1])),
                   'monthly': float(np.mean(scores[2])),
                   'others': float(np.mean(scores[3])),
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

def pre_process_m3(LOAD_M3, fpath_data, frequency, subset,
                   fpath_true_train, fpath_true_test,
                   fpath_max_scale_train, fpath_max_scale_test,
                   fpath_train_scaling,
                   fpath_test_scaling
                   ):
    if LOAD_M3:
        m3dataset = M3Dataset(fpath_dataset_file=fpath_data)
        train_data_m3 = m3dataset.load(frequency, subset, True)
        test_data_m3 = m3dataset.load(frequency, subset, False)

        scale_train_data_m3 = m3dataset.scale(train_data_m3)
        scale_test_data_m3 = m3dataset.scale(test_data_m3)

        true_train, true_test = [], []
        scale_train, scale_test = [], []
        scaling_factor_train, scaling_factor_test = [], []

        for ii in train_data_m3:
            true_train.append(ii['target'])

        for ii in test_data_m3:
            true_test.append(ii['target'])

        for ii in scale_train_data_m3:
            scale_train.append(ii['target'])
            scaling_factor_train.append(ii['scaling_factor'])

        for ii in scale_test_data_m3:
            scale_test.append(ii['target'])
            scaling_factor_test.append(ii['scaling_factor'])

        true_train = np.array(true_train, dtype=object)
        true_test = np.array(true_test, dtype=object)
        scale_train = np.array(scale_train, dtype=object)
        scale_test = np.array(scale_test, dtype=object)

        np.save(fpath_true_train+'.npy', true_train)
        np.save(fpath_true_test+'.npy', true_test)
        np.save(fpath_max_scale_train + '.npy', scale_train)
        np.save(fpath_max_scale_test + '.npy', scale_test)
        np.save(fpath_train_scaling + '.npy', scaling_factor_train)
        np.save(fpath_test_scaling + '.npy', scaling_factor_test)
    else:
        true_train = np.load(fpath_true_train+'.npy', allow_pickle=True)
        true_test = np.load(fpath_true_test+'.npy', allow_pickle=True)
        scale_train = np.load(fpath_max_scale_train + '.npy', allow_pickle=True)
        scale_test = np.load(fpath_max_scale_test + '.npy', allow_pickle=True)
        scaling_factor_train = np.load(fpath_train_scaling + '.npy', allow_pickle=True)
        scaling_factor_test = np.load(fpath_test_scaling + '.npy', allow_pickle=True)
    return true_train, true_test, scale_train, scale_test, scaling_factor_train, scaling_factor_test



if __name__ == '__main__':
    # ===-----------------------------------------------------------------------===
    # Argument parsing
    # ===-----------------------------------------------------------------------===
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", default="M", type=str, dest="frequency", help="frequency")
    parser.add_argument("--fpath_ori_data", default="", type=str, dest="fpath_ori_data", help="file path for original data")
    parser.add_argument("--Load_m3", default="True", type=bool, dest="Load_m3", help="if read original data")
    parser.add_argument("--subset", default="monthly", type=str, dest="subset", help="yearly | monthly | quarterly | others")
    parser.add_argument("--fpath_true_train", default="", type=str, dest="fpath_true_train", help="file path for true train data")
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
    parser.add_argument("--train_index_known", default="True", type=bool, dest="train_index_known", help="if data index known")

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
    pre_process_m3(LOAD_M3=options.Load_m3, fpath_data=options.fpath_ori_data,
                   frequency=options.frequency, subset=options.sun_set,
                   fpath_true_train=options.fpath_true_train, fpath_true_test=options.fpath_true_test,
                   fpath_max_scale_train=options.fpath_max_scale_train, fpath_max_scale_test=options.fpath_max_scale_test,
                   fpath_train_scaling=options.fpath_train_scaling,
                   fpath_test_scaling=options.fpath_test_scaling
                   )

    get_dataset(fpath_true_train=options.fpath_true_train,  fpath_true_test=options. fpath_true_test,
                fapth_scale_train=options.fapth_scale_train, fpath_target_train=options.fpath_target_train,
                fpath_target_mask=options.fpath_target_mask, fpath_target_train_scale=options.fpath_target_train_scale,
                fpath_target_mask_scale=options.fpath_target_mask_scale, fpath_train_index=options.fpath_train_index,
                fpath_mask_index=options.fpath_mask_index, ratio=options.ratio, train_index_known=options.train_index_known )

    get_split_dataset(fpath_target_train=options.fpath_target_train, fpath_target_mask=options.fpath_target_mask,
                      fpath_target_train_scale=options.fpath_target_train_scale,
                      fpath_target_mask_scale=options.fpath_target_mask_scale,

                      fpath_target_train_save=options.fpath_split_target_train,
                      fpath_target_mask_save=options.fpath_split_target_mask,
                      fpath_scale_target_train_save=options.fpath_split_scale_target_train,
                      fpath_scale_target_mask_save=options.fpath_split_scale_target_mask,
                      seq_len=options.seq_len)





