import csv
import numpy as np
import pandas as pd


def save_csv_index(file_name, data):
    f = open(file_name+'.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(data)


def save_csv(file_name, data):
    f = open(file_name+'.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)

    for i in range(data.shape[0]):
        #print(data[i])
        csv_writer.writerow(data[i])


def read_csv(file_name):
    f = open(file_name+'.csv', 'r')
    csv_reader = np.loadtxt(f, delimiter=',')
    data = []
    for item in csv_reader:
        # print(item)
        data.append(np.array(item))

    data = np.array(data)
    return data


def read_csv_varying_length(file_name):
    file_name = file_name+'.csv'
    largest_column_count = 0
    with open(file_name, 'r') as temp_f:
        lines = temp_f.readlines()
        for l in lines:
            column_count = len(l.split(',')) + 1
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count
    temp_f.close()
    column_names = [i for i in range(0, largest_column_count)]
    df = pd.read_csv(file_name, header=None, delimiter=',', names=column_names)

    df = df.stack().groupby(level=0).apply(list)
    return df



def get_dataset(fpath_true_train, fpath_true_test,
                fapth_scale_train, fpath_target_train, fpath_target_mask,
                fpath_target_train_scale, fpath_target_mask_scale,
                fpath_train_index, fpath_mask_index,
                ratio=1.0, train_index_known=False):

    true_train = np.load(fpath_true_train+'.npy', allow_pickle=True)
    scale_train = np.load(fapth_scale_train+'.npy', allow_pickle=True)
    true_test = np.load(fpath_true_test+'.npy', allow_pickle=True)
    ratio_str = int(100*ratio)

    if train_index_known:
        train_index = read_csv(fpath_train_index)
        if ratio < 1.0:
            mask_index = read_csv(fpath_mask_index)
    else:
        train_index_num = int(ratio * len(true_train))
        train_index = np.random.permutation(len(true_train))[:train_index_num]
        if ratio < 1.0:
            mask_index = np.random.permutation(len(true_train))[train_index_num:]
            save_csv_index(fpath_mask_index+'_ratio'+str(ratio_str), mask_index)
            mask_index = mask_index.astype(int)

        save_csv_index(fpath_train_index+'_ratio'+str(ratio_str), train_index)

    train_index = train_index.astype(int)

    target_train_data = true_train[train_index-1]
    scale_target_train_data = scale_train[train_index-1]

    if ratio < 1.0:
        target_mask_data = true_train[mask_index-1]
        scale_target_mask_data = scale_train[mask_index-1]
        save_csv(fpath_target_mask+'_ratio'+str(ratio_str), target_mask_data)
        save_csv(fpath_target_mask_scale+'_ratio'+str(ratio_str), scale_target_mask_data)

    save_csv(fpath_target_train+'_ratio'+str(ratio_str), target_train_data)
    save_csv(fpath_target_train_scale+'_ratio'+str(ratio_str), scale_target_train_data)


def split_data(ori_data, seq_len):
    no = len(ori_data)
    temp_data = []
    # Cut data by sequence length
    for i in range(no):
        x_i = ori_data[i]
        for j in range(len(x_i)-seq_len):
            x_ = x_i[j:j + seq_len]
            x_ = np.append(x_, int(i))

            if len(x_) < 2:
               x_ = x_[:, np.newaxis]

            temp_data.append(x_)

    return np.array(temp_data)


def get_split_dataset(fpath_target_train, fpath_target_mask,
                      fpath_target_train_scale, fpath_target_mask_scale,
                      fpath_target_train_save, fpath_target_mask_save,
                      fpath_scale_target_train_save, fpath_scale_target_mask_save, seq_len):

    target_train_data = read_csv_varying_length(fpath_target_train)
    target_mask_data = read_csv_varying_length(fpath_target_mask)
    scale_target_train_data = read_csv_varying_length(fpath_target_train_scale)
    scale_target_mask_data = read_csv_varying_length(fpath_target_mask_scale)

    split_scale_target_train_data = split_data(scale_target_train_data, seq_len)
    split_scale_target_mask_data = split_data(scale_target_mask_data, seq_len)

    split_target_train_data = split_data(target_train_data, seq_len)
    split_target_mask_data = split_data(target_mask_data, seq_len)

    save_csv(fpath_target_train_save, split_target_train_data)
    save_csv(fpath_target_mask_save, split_target_mask_data)
    save_csv(fpath_scale_target_train_save, split_scale_target_train_data)
    save_csv(fpath_scale_target_mask_save, split_scale_target_mask_data)

    print('split_target_train_data:', split_target_train_data.shape)
    print('split_target_mask_data:', split_target_mask_data.shape)
    print('split_scale_target_train_data:', split_scale_target_train_data.shape)
    print('split_scale_target_mask_data:', split_scale_target_mask_data.shape)

