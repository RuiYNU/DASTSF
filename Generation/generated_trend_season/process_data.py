import pandas as pd
import pickle
import sys
sys.path.append(".")
from Generation.generated_trend_season.processor import Processor
from fastNLP import DataSet
import csv
import numpy as np
import os
import errno
import argparse
import torch


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def save_csv(file_name, data):
    f = open(file_name+'.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)

    for i in range(data.shape[0]):
        csv_writer.writerow(data[i])


def read_csv(file_name):
    f = open(file_name+'.csv', 'r')
    csv_reader = np.loadtxt(f, delimiter=',')
    data = []
    for item in csv_reader:
        data.append(np.array(item))

    data = np.array(data)
    return data


def pre_process(data_name, seq_len, file_name_train, file_name_mask):
    target_train_data_ = read_csv(file_name_train)
    target_mask_data_ = read_csv(file_name_mask)

    # in original data, the last column is not data value
    # ----------------------------------------------------------
    # target_train_data_ = target_train_data_[:, :-1]
    # target_mask_data_ = target_mask_data_[:, :-1]
    # -----------------------------------------------------------

    P = Processor()
    P_mask = Processor()

    target_train_data = pd.DataFrame(target_train_data_) # limited target training data (for training)
    target_mask_data = pd.DataFrame(target_mask_data_) # masked target data (only for validation)

    # min_max_scalar normalization
    normalize_target_train_data = P.fit_transform(target_train_data)
    normalize_target_mask_data = P_mask.fit_transform(target_mask_data)

    # save target re_min_max factor for inverse_transform of generated data
    min_scale_dict = {}
    min_scale_dict['min'] = []
    min_scale_dict['max'] = []
    for model in P.models:
        min_scale_dict['min'].append(model.model.min_)
        min_scale_dict['max'].append(model.model.scale_)

    min_scale_dict_mask = {}
    min_scale_dict_mask['min'] = []
    min_scale_dict_mask['max'] = []
    for model in P_mask.models:
        min_scale_dict_mask['min'].append(model.model.min_)
        min_scale_dict_mask['max'].append(model.model.scale_)
    # save target_mask re_min_max factor for inverse_transform of target_mask data

    if len(normalize_target_train_data.shape) < 3:
        normalize_target_train_data = normalize_target_train_data[:, :, np.newaxis]
        normalize_target_mask_data = normalize_target_mask_data[:, :, np.newaxis]
        target_train_data_ = target_train_data_[:, :, np.newaxis]
        target_mask_data_ = target_mask_data_[:, :, np.newaxis]

    dataset = DataSet({"seq_len": [seq_len] * len(normalize_target_train_data),
                       "dyn": normalize_target_train_data})

    dic = {"train_set": dataset,
           "processor": P,
           "normalize_train_data": normalize_target_train_data,
           "normalize_mask_data": normalize_target_mask_data,
           "train_data": target_train_data_,
           "mask_data": target_mask_data_,
           "train_scale_dict": min_scale_dict,
           "mask_scale_dict": min_scale_dict_mask
           }

    make_sure_path_exists("./Generation/generated_trend_season/data")
    with open("./Generation/generated_trend_season/data/{}.pkl".format(data_name), "wb") as f:
        pickle.dump(dic, f)


if __name__ == '__main__':
    # step 1: read_data from m3/tourism/traffic (reference:N-Beats) and use sliding-window to split data
    #         randomly select target_train_data from original train data, the rest is used as target_mask data.
    #         target_mask_data only used for validation

    # step 2: python pre_process.py --data_name XX --file_target_train XXX --file_target_mask XXX



    # ===-----------------------------------------------------------------------===
    # Argument parsing
    # ===-----------------------------------------------------------------------===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    # parser.add_argument("--devi", default="0", dest="devi", help="gpu")
    parser.add_argument("--file_target_train", default='target_data', type=str, help="normalize target data path")
    parser.add_argument("--file_target_mask", default='target_mask_data', type=str, help="normalize target mask data path")
    parser.add_argument("--data_name", default='m3', type=str, help="dataset name")
    parser.add_argument("--seq_len", default=24, type=int, help="seq_len")

    options = parser.parse_args()
    # options.seq_len = 8
    # options.file_target_train = './Generation/generated_trend_season/data/tourism_yearly_train_data'
    # options.file_target_mask = './Generation/generated_trend_season/data/tourism_yearly_mask_data'
    # options.data_name = 'tourism_yearly'

    pre_process(data_name=options.data_name, seq_len=options.seq_len,
                file_name_train=options.file_target_train,
                file_name_mask=options.file_target_mask)







