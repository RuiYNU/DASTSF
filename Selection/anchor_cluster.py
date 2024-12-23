import sys
sys.path.append(".")
import torch
import torch.nn.functional as F
from fastNLP import DataSet
from sys import maxsize
import pandas as pd
import pickle
import random
import argparse
import csv
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from annoy import AnnoyIndex
from sklearn.manifold import TSNE
from Generation.generated_trend_season.train_aegan import AeGAN
# from Generation.generated_trend_season.test_aegan import AeGAN
from Selection.tri_distance import *
from Selection.anchor_loss import *


from Selection.anchor_trainer import Trainer, FeaModel


from Generation.generated_trend_season.process_data import *
from Generation.main_generation import read_syn_dataset, re_scale
from Generation.generated_trend_season.visualization import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_syn_fea(options,  save_fea):

    if save_fea:

        # ===-----------------------------------------------------------------------===
        # Read in dataset
        # ===-----------------------------------------------------------------------===
        dyn_processor, train_set, train_scale_dict, mask_scale_dict, \
        normalize_target_train_data, normalize_target_mask_data, \
        target_train_data, target_mask_data = read_syn_dataset(options)

        params = vars(options)
        root_dir = "{}/{}".format(options.root_dir, options.task_name)
        params["root_dir"] = root_dir
        params["seq_len"] = options.seq_len
        params["device"] = options.device
        re_mask_data = re_scale(normalize_target_mask_data, mask_scale_dict)

        syn = AeGAN(processors=dyn_processor, params=params)
        # 1111syn = AeGAN((None, dyn_processor), params)

        if options.training_ae:
            syn.train_ae(train_set, options.ae_epochs)
            res, h = syn.eval_ae(train_set)
            with open("{}/hidden".format(root_dir), "wb") as f:
                pickle.dump(h, f)
            syn.train_gan(train_set, options.gan_epochs, options.d_update)
        else:
            options.fix_ae = '{}/ae_new2.dat'.format(params["root_dir"])
            options.fix_gan = '{}/generator_new2.dat'.format(params["root_dir"])
            syn.load_ae(options.fix_ae)
            print('AE Load')
            syn.load_generator(options.fix_gan)
            print('GAN Load')

        # ===-----------------------------------------------------------------------===
        # Construct mask/source DataSet
        target_mask_set = DataSet({"seq_len": [params["seq_len"]] * len(normalize_target_mask_data),
                                   "dyn": normalize_target_mask_data})
        target_mask_set.set_input("dyn", "seq_len")

        normalize_source_data = read_csv(file_name=options.fpath_normalize_src_data)
        normalize_source_data = normalize_source_data.reshape(normalize_source_data.shape[0],
                                                              params["seq_len"], -1)
        source_set = DataSet({"seq_len": [params["seq_len"]] * len(normalize_source_data),
                              "dyn": normalize_source_data})
        source_set.set_input("dyn", "seq_len")
        # ===---------------------------------------------------------------------------------== #

        # ==------------------------------------- features------------------------------------== #
        _, normalize_target_train_fea = syn.eval_ae(train_set)
        _, normalize_target_mask_fea = syn.eval_ae(target_mask_set)
        _, normalize_source_fea = syn.eval_ae(source_set)

        # ==------------------------------------- save csv file--------------------------------== #
        save_csv(options.fpath_normalize_train_data, normalize_target_train_data.reshape(normalize_target_train_data.shape[0], -1))
        save_csv(options.fpath_normalize_mask_data, normalize_target_mask_data.reshape(normalize_target_mask_data.shape[0],-1))
        save_csv(options.fpath_normalize_train_fea, normalize_target_train_fea)
        save_csv(options.fpath_normalize_mask_fea, normalize_target_mask_fea)

        save_csv(options.fpath_normalize_src_fea, normalize_source_fea)

        print('# --------------------------------AutoEncoder loaded------------------------------------#')
        print('target_features:', normalize_target_train_fea.shape,
              'source_features:', normalize_source_fea.shape,
              'target_mask_features:', normalize_target_mask_fea.shape)
    else:
        target_train_data = read_csv(options.fpath_train_data)
        target_mask_train_data = read_csv(options.fpath_mask_data)
        normalize_target_train_data = read_csv(options.fpath_normalize_train_data)
        normalize_target_mask_data = read_csv(options.fpath_normalize_mask_data)
        normalize_target_train_fea = read_csv(options.fpath_normalize_train_fea)
        normalize_target_mask_fea = read_csv(options.fpath_normalize_mask_fea)
        normalize_source_data = read_csv(options.fpath_normalize_src_data)
        normalize_source_fea = read_csv(options.fpath_normalize_src_fea)

        if len(target_train_data.shape) < 3:
            target_train_data = target_train_data.reshape((target_train_data.shape[0], options.seq_len, -1))
            target_mask_train_data = target_mask_train_data.reshape((target_mask_train_data.shape[0],
                                                                     options.seq_len, -1))
            normalize_target_train_data = normalize_target_train_data.reshape((normalize_target_train_data.shape[0],
                                                                               options.seq_len, -1))
            normalize_target_mask_data = normalize_target_mask_data.reshape((normalize_target_mask_data.shape[0],
                                                                             options.seq_len, -1))
            normalize_source_data = normalize_source_data.reshape((normalize_source_data.shape[0],
                                                                   options.seq_len, -1))

        print('target_target_train_data:', target_train_data.shape)
        print('target_target_mask_data:', target_mask_train_data.shape)
        print('normalize_target_train_data:', normalize_target_train_data.shape)
        print('normalize_target_mask_data:', normalize_target_mask_data.shape)
        print('normalize_source_data:', normalize_source_data.shape)
        print('normalize_source_features:', normalize_source_fea.shape)

    return normalize_target_train_fea, normalize_target_train_data,\
           normalize_source_fea, normalize_source_data, \
           normalize_target_mask_data, normalize_target_mask_fea


class Cluster(object):
    def __init__(self, args, train_set):
        self.args = args
        self.train_set = train_set
        self.old_indices = range(len(self.train_set))

        self.kmeans = KMeans(n_clusters=args.cluster)
        self.num_cluster = self.args.cluster

    def kmeanscluster(self):
        labels = self.kmeans.fit_predict(self.train_set)
        centers = self.kmeans.cluster_centers_  # n_cluster * dim
        return labels, centers

    def get_labels_centers(self):
        labels, centers = self.kmeanscluster()
        return labels, centers



def create_dataloader(params, fea_train, fea_source, target_labels, index):

    # fea_train: N * dim
    sampler = AnchorSampler(data_source=fea_source, data_target=fea_train, labels=target_labels,
                            k=params['n_neighbors'], index_sort=index)


    train_loader = DataLoader(AnchorPreprocessor(target_dataset=fea_train, source_dataset=fea_source, target_labels=target_labels),
                              batch_size=params['tri_batch_size'], sampler=sampler
    )
    # a_feat, p_feat, n_feat, index(i, pos_ind, pos_pos, neg_ind, neg_pos)

    return train_loader


def cal_distance_tgt_src(X_tgt, X_src):
    n_tgt, dim = X_tgt.shape
    n_src, dim = X_src.shape

    tgt_src_dist = np.empty((n_tgt, n_src))
    for i in range(n_tgt):
        tgt_fea_i = torch.Tensor(X_tgt[i])
        if len(tgt_fea_i.shape) < 2:
            tgt_fea_i = tgt_fea_i[np.newaxis, :]

        tgt_src_dist[i] = compute_correspond_dist(tgt_fea_i, torch.from_numpy(X_src),
                                                  dist_type='cosine').cpu().detach().numpy()

    return tgt_src_dist # distance


def cal_distance(X, X_src=None, distance="angular", num_trees=5):
    # X: X_tgt_n * dim  X_src: X_src_n * dim
    # return: X_tgt * X_src
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if len(X_src.shape) > 2:
        X_src = X_src.reshape(X_src.shape[0], -1)

    print('NUM_TREE:', num_trees, 'SCR:', X_src.shape, X.shape)
    n, dim = X.shape
    if X_src is None:
        n_extra = n
        X_src = X
    else:
        n_extra = X_src.shape[0]

    tree = AnnoyIndex(dim, metric=distance)
    for i in range(n_extra):
        tree.add_item(i, X_src[i, :])


    tree.build(num_trees)

    # ==-----------------------------------------------------------------------------------------== #
    nn_index = np.empty((n, n_extra)) # (n_tgt * n_src: sorted by dist)
    distance = np.empty((n, n_extra))
    for i in range(n):
        x_tgt_i = X[i, :]
        nn_index[i, :], distance[i, :] = tree.get_nns_by_vector(vector=x_tgt_i, n=n_extra, include_distances=True)
        dist_i = np.square(distance[i, :]) *1./ 2 # Annoy cosine-distance: sqrt(2(1-cos))
        distance[i, :] = dist_i
        nn_index[i, :] = np.array(nn_index[i,:])


    print('Finish Tree')

    return nn_index.astype(int), distance


def train_cluster(options, save_fea):
    normalize_target_train_fea, normalize_target_train_data, \
    normalize_source_fea, normalize_source_data, \
    normalize_target_mask_data, normalize_target_mask_fea = load_syn_fea(options, save_fea)

    N_tgt_samples = normalize_target_train_fea.shape[0]
    tgt_fea_list, src_fea_list = normalize_target_train_fea, normalize_source_fea

    if len(tgt_fea_list.shape) < 3:
        hid_dim = 1
    else:
        hid_dim = tgt_fea_list.shape[-1]

    fea_dim = tgt_fea_list.shape[-1]
    if options.read_dist:
        # D_tgt_src = read_csv(options.fpath_dist_src_tgt)
        index_sort = read_csv(options.fpath_dist_src_tgt_index)
        D_tgt_src_sort = read_csv(options.fpath_dist_src_tgt_sort)
        index_sort = index_sort.astype(int)

    else:
        print('# -------------------------------------Begin calculate src-tgt distance--------------------------#')
        index_sort, D_tgt_src_sort = cal_distance(tgt_fea_list, src_fea_list)
        # index_sort, D_tgt_src_sort = cal_distance(normalize_target_train_data, normalize_source_data)
        print('Finish Distance Calculation')


        save_csv(options.fpath_dist_src_tgt_sort, D_tgt_src_sort)
        save_csv(options.fpath_dist_src_tgt_index, index_sort)

    print('# -------------------------------------Finish src-tgt distance calculation--------------------------#')


    # ==--------------------------------------------------Cluster---------------------------------------== #
    if options.run_cluster:
        print('# -----------------------------------------Begin Cluster-----------------------------------')
        cluster = Cluster(args=options, train_set=normalize_target_train_fea)

        labels, centers = cluster.get_labels_centers()
        save_csv(options.fpath_cluster_labels, labels.reshape(len(labels), 1))
        save_csv(options.fpath_cluster_centers, centers)
        print('# ---------------------------------------Finish Cluster------------------------------------')
    else:
        labels = read_csv(options.fpath_cluster_labels)
        centers = read_csv(options.fpath_cluster_centers)

    print('CLUSTER_END! Labels:{}, Centers:{}'.format(labels.shape, centers.shape))

    # ==---------------------------------------------Trainer---------------------------------------------== #
    params = vars(options)
    trainer = Trainer(params=params, input_dims=hid_dim, fea_dims=fea_dim,
                      hidden_dims=params["hidden_dim"], layers=params["layers"],
                      N_tgt_samples=N_tgt_samples, D_tgt_src_sort=D_tgt_src_sort,
                      dropout=0.1, alpha=0.5, print_inter=5, device=DEVICE)


    target_loader = create_dataloader(params, fea_train=tgt_fea_list, fea_source=src_fea_list,
                                      target_labels=labels, index=index_sort)

    trainer.train_epoch(target_loader, epoch=params['cluster_epochs'])

# ==-----------------------------------------------------EVAL-----------------------------------------------== #
class InferData(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        tgt_fea = self.data[index]
        tgt_fea_tensor = torch.tensor(np.array(tgt_fea)).to(DEVICE)
        return tgt_fea_tensor

    def __len__(self):
        return len(self.data)


def eval_model(options):
    params = vars(options)

    tgt_fea_list = read_csv(options.fpath_normalize_train_fea)
    src_fea_list = read_csv(options.fpath_normalize_src_fea)
    tgt_input, src_input = tgt_fea_list, src_fea_list

    src_len, tgt_len = len(src_fea_list), len(tgt_fea_list)

    if len(tgt_fea_list.shape) < 3:
        hid_dim = 1
    else:
        hid_dim = tgt_fea_list.shape[-1]

    fea_dim = tgt_fea_list.shape[-1]

    # ==-------------------------------------------------load fea_model---------------------------------------------== #

    fea_model = FeaModel(params=params, input_dims=hid_dim, fea_dims=fea_dim,
                         hidden_dims=params["hidden_dim"], layers=params["layers"],
                         dropout=0.1, alpha=0.5, device=DEVICE)
    root_dir = params['model_save_path']
    fea_model.load_state_dict(torch.load(root_dir + '/fea_model_' + str(options.epoch_train) + '.dat', map_location=DEVICE))
    fea_model = fea_model.to(DEVICE)
    fea_model.eval()
    # ==-------------------------------------------------load fea_model---------------------------------------------== #

    if options.save_src_tgt_score:
        src_dataset = InferData(src_input)
        src_dataloader = DataLoader(dataset=src_dataset, batch_size=params['fea_batch_size'],
                                    shuffle=False, drop_last=False)

        tgt_dataset = InferData(tgt_input)
        tgt_dataloader = DataLoader(dataset=tgt_dataset, batch_size=params['fea_batch_size'],
                                    shuffle=False, drop_last=False)

        for i, data in enumerate(src_dataloader):
            src_data = data
            if len(data.shape) < 3:
                src_data = data[:, :, np.newaxis]

           #############################################
            src_data = src_data.float()
            #############################################
            src_t = fea_model.encoder_q(src_data)
            if i == 0:
                src_fea_encode = src_t
            else:
                src_fea_encode = torch.cat((src_fea_encode, src_t))

        for i, data in enumerate(tgt_dataloader):
            tgt_data = data
            if len(data.shape) < 3:
                tgt_data = data[:, :, np.newaxis]

                ####################################################
            tgt_data = tgt_data.float()
                ####################################################
            tgt_t = fea_model.encoder_q(tgt_data)

            if i == 0:
                tgt_fea_encode = tgt_t
            else:
                tgt_fea_encode = torch.cat((tgt_fea_encode, tgt_t))

        print('tgt:', tgt_fea_encode.shape)
        print('src:', src_fea_encode.shape)

        src_tgt_score = np.empty((src_len, tgt_len))
        for i in range(src_len):
            src_fea_i = src_fea_encode[i]
            if len(src_fea_i.shape) < 2:
                src_fea_i = src_fea_i[np.newaxis, :]
            src_tgt_score[i] = F.cosine_similarity(src_fea_i, tgt_fea_encode).cpu().detach().numpy()

        save_csv(options.fpath_src_tgt_score, src_tgt_score)
    else:
        src_tgt_score = read_csv(options.fpath_src_tgt_score)

    if options.save_mean_src_tgt_score:
        mean_src_tgt_score = []
        for i in range(len(src_tgt_score)):
            one_infer_res = src_tgt_score[i]
            mean_v = np.mean(one_infer_res)
            mean_src_tgt_score.append([i, mean_v])
        save_csv(options.fpath_mean_src_tgt_score, np.array(mean_src_tgt_score))
    else:
        mean_src_tgt_score = read_csv(options.fpath_mean_src_tgt_score)


def plot_fea(options):
    src_fea_list = read_csv(options.fpath_normalize_src_fea)
    tgt_mask_fea_list = read_csv(options.fpath_normalize_mask_fea) # only for validation

    src_tgt_score_mean = read_csv(options.fpath_mean_src_tgt_score)

    sort_index = np.argsort(-src_tgt_score_mean[:, -1])  # max-->min
    select_num = int(options.ratio * len(src_tgt_score_mean[:, -1]))
    src_tgt_mean_index = sort_index[:select_num]
    no_src_tgt_mean_index = sort_index[select_num:]

    print('src_tgt_max_index:', src_tgt_score_mean.shape, src_tgt_mean_index.shape)
    print('src_fea_list:', src_fea_list.shape, 'select_num:', select_num, 'src_tgt_mean_index:',
          len(src_tgt_mean_index))
    select_src_tgt = src_fea_list[src_tgt_mean_index]
    no_select_src_tgt = src_fea_list[no_src_tgt_mean_index]

    if len(tgt_mask_fea_list.shape) < 3:
        tgt_mask_fea_list = tgt_mask_fea_list[:, :, np.newaxis]
        no_select_src_tgt = no_select_src_tgt[:, :, np.newaxis]
        select_src_tgt = select_src_tgt[:, :, np.newaxis]

    print('tgt_mask_fea_list:', tgt_mask_fea_list.shape, 'src_fea_list:', select_src_tgt.shape,
          'no_src_fea_list:', no_select_src_tgt.shape)

    select_src_tgt_copy = select_src_tgt
    no_select_src_tgt_copy = no_select_src_tgt
    if len(select_src_tgt_copy.shape) > 2:
        select_src_tgt_copy = select_src_tgt_copy.reshape((select_src_tgt_copy.shape[0], -1))
        no_select_src_tgt_copy = no_select_src_tgt_copy.reshape((no_select_src_tgt_copy.shape[0], -1))

    save_csv(options.fpath_select_src_data, select_src_tgt_copy)
    save_csv(options.fpath_no_select_src_data, no_select_src_tgt_copy)

    print('==-------------------------------Finish saving selected source data--------------------------------==')
    pca_and_tsne_new(ori_ts=np.array(tgt_mask_fea_list),
                     art_ts=np.array(no_select_src_tgt),
                     sel_ts=np.array(select_src_tgt),
                     experiment_name=options.task_name,
                     pics_dir=options.model_save_path,
                     label_ori='masked_target', label_syn='rejected_src',
                     label_selected_src='selected_src',
                     color_alpha=0.2, use_legend=False, fontsize=18)


if __name__ == '__main__':
    # ===-----------------------------------------------------------------------===
    # Argument parsing
    # ===-----------------------------------------------------------------------===
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default="tourism_yearly", help="task_name")
    parser.add_argument("--source_dataset", default="./Selection/pre_tourism/data/source_data", dest="source_dataset", help=".pkl file to use")
    parser.add_argument("--dataset", default="./Generation/generated_trend_season/data/tourism_yearly.pkl", dest="dataset",
                        help=".pkl file to use")
    parser.add_argument("--device", default="0", dest="devi", help="gpu")

    # ===------------------------------------------------------------------------===
    # Argument for syn
    # ==-------------------------------------------------------------------------===
    parser.add_argument("--root_dir", default="./Generation/generated_trend_season/result", dest="root_dir",
                        help="Directory for models")
    parser.add_argument("--ae-epochs", default=2000, dest="ae_epochs", type=int,
                        help="training epochs for autoencoder")
    parser.add_argument("--gan-epochs", default=6000, dest="gan_epochs", type=int,
                        help="training epochs for WGAN")
    parser.add_argument("--d-update", default=5, dest="d_update", type=int,
                        help="discriminator updates per generator update")

    parser.add_argument("--dim", default=1, dest="dim", type=int, help="sequence_dim")

    parser.add_argument("--gan-lr", default=1e-4, dest="gan_lr", type=float, help="WGAN learning rate")
    parser.add_argument("--gan-alpha", default=0.99, dest="gan_alpha", type=float, help="for RMSprop")
    parser.add_argument("--noise-dim", default=96, dest="noise_dim", type=int, help="dim of WGAN noise state")

    parser.add_argument("--embed-dim", default=96, dest="embed_dim", type=int, help="dim of embedding")
    parser.add_argument("--hidden-dim", default=24, dest="hidden_dim", type=int, help="dim of GRU hidden state")
    parser.add_argument("--layers", default=3, dest="layers", type=int, help="layers")
    parser.add_argument("--ae-lr", default=1e-3, dest="ae_lr", type=float, help="learning rate of autoencoder")
    parser.add_argument("--weight-decay", default=0, dest="weight_decay", type=float, help="weight decay")
    parser.add_argument("--scale", default=1, dest="scale", type=float, help="if scale")
    parser.add_argument("--dropout", default=0.0, dest="dropout", type=float)
    parser.add_argument("--ae-batch-size", default=128, dest="ae_batch_size", type=int, help="batchsize for AE")
    parser.add_argument("--gan-batch-size", default=128, dest="gan_batch_size", type=int, help="batchsize for GAN")
    parser.add_argument("--training-ae", default=False, dest="training_ae", type=bool, help='train AeGAN')
    parser.add_argument("--epochs", default=2000, dest="epochs", type=int,
                        help="Training epoch for autoencoder")
    # ==----------------------------------------------------------------------------------------------------===
    # Argument for anchor_trainer
    # ==----------------------------------------------------------------------------------------------------===
    parser.add_argument("--seq_len", default=8, dest="seq_len", type=int, help="seq_len")
    parser.add_argument("--tri_layers", default=2, dest="tri_layers", type=int, help="tri_layers")
    parser.add_argument("--tri_batch_size", default=128, dest="tri_batch_size", type=int,
                        help="batch size for WGAN")

    parser.add_argument("--fea-batch-size", default=128, dest="fea_batch_size", type=int,
                        help="Minibatch size for fea")

    parser.add_argument("--dist_type", default='cosine', dest="dist_type", type=str)
    parser.add_argument("--triloss_mean", default=True, dest="triloss_mean", type=bool)
    parser.add_argument("--margin", default=1, dest="margin", type=float)
    parser.add_argument('--weight_temp', default=0.5, dest="weight_temp", type=float)
    parser.add_argument('--n_neighbors', default=10, dest="n_neighbors", type=int)

    parser.add_argument('--epoch_train', default=1000, dest="epoch_train", type=int)
   # ==-------------------------------------------------------------------------------------------------------------===

    parser.add_argument('--model_save_path', default='./Selection/src_tgt_model/model_save', dest="model_save_path",
                        type=str)
    parser.add_argument('--layers_tri', default=1, dest="layers_tri", type=int)
    parser.add_argument('--fpath_normalize_train_fea', default='Generation/generated_trend_season/data/tourism_yearly_normalize_train_fea',
                        dest="fpath_normalize_train_fea", type=str)

    parser.add_argument('--fpath_normalize_train_data', default='Generation/generated_trend_season/data/tourism_yearly_normalize_train_data',
                        dest="fpath_normalize_train_data", type=str)

    parser.add_argument('--fpath_normalize_mask_fea', default='Generation/generated_trend_season/data/tourism_yearly_normalize_mask_fea',
                        dest="fpath_normalize_mask_fea", type=str)

    parser.add_argument('--fpath_normalize_mask_data', default='Generation/generated_trend_season/data/tourism_yearly_normalize_mask_data',
                        dest="fpath_normalize_mask_data", type=str)

    parser.add_argument('--fpath_normalize_src_data', default='Generation/generated_trend_season/data/tourism_yearly_src_data',
                        dest="fpath_normalize_src_data", type=str)
    parser.add_argument('--fpath_normalize_src_fea', default='Generation/generated_trend_season/data/tourism_yearly_src_fea',
                        dest="fpath_normalize_src_fea", type=str)

    parser.add_argument('--fpath_train_data', default='Generation/generated_trend_season/data/tourism_yearly_train_data',
                        dest="fpath_train_data", type=str)
    parser.add_argument('--fpath_mask_data', default='Generation/generated_trend_season/data/tourism_yearly_mask_data',
                        dest="fpath_mask_data", type=str)


    # ==-------------------------------------------------------------------------------------------------------------===
    # Argument for selected_src_data
    parser.add_argument('--save_mean_src_tgt_score', default=True, dest="save_mean_src_tgt_score", type=bool)
    parser.add_argument('--save_src_tgt_score', default=True, dest="save_src_tgt_score", type=bool)

    parser.add_argument('--fpath_mean_src_tgt_score', default='./Selection/src_tgt_model/model_save/selected_src/mean_src_tgt_score',
                        dest="fpath_mean_src_tgt_score", type=str)

    parser.add_argument('--fpath_select_src_data', default='./Selection/src_tgt_model/model_save/selected_src/select_src_data',
                        dest='fpath_select_src_data', type=str)
    parser.add_argument('--fpath_no_select_src_data', default='./Selection/src_tgt_model/model_save/selected_src/no_select_src_data',
                        dest='fpath_no_select_src_data', type=str)
    parser.add_argument('--fpath_src_tgt_score', default='./Selection/src_tgt_model/model_save/selected_src/src_tgt_score',
                        dest="fpath_src_tgt_score", type=str)

    parser.add_argument('--ratio', default=0.4, dest='ratio', type=float)
    # ==-------------------------------------------------------------------------------------------------------------===


    parser.add_argument('--read-dist', default=False, dest="read_dist", type=bool)

    # ==-------------------------------------------------------------------------------------------------------------===
    # Argument for contrast
    parser.add_argument('--temperature', default=0.1, dest="temperature", type=float)
    parser.add_argument('--contrast_mode', default='one', dest="contrast_mode", type=str)
    parser.add_argument('--base_temperature', default=0.1, dest="base_temperature", type=float)
    # ==-------------------------------------------------------------------------------------------------------------===

    # Argument for cluster
    parser.add_argument('--rho', type=float, default=0.05, dest='rho',
                        help="rho percentage, default: 1.6e-3")
    parser.add_argument('--cluster', default=5, dest="cluster", type=int)
    parser.add_argument('--mining_ratio', default=0.2, dest="mining_ratio", type=float)
    parser.add_argument("--cluster-epochs", default=1000, dest="cluster_epochs", type=int,
                        help="Training epoch for cluster")

    parser.add_argument("--run-cluster", default=False, dest="run_cluster", type=bool,
                        help="if run_cluster")

    parser.add_argument('--fpath_cluster_labels', default='Selection/src_tgt_model/cluster/cluster_labels',
                        help="path for cluster_labels", type=str)
    parser.add_argument('--fpath_cluster_centers', default='Selection/src_tgt_model/cluster/cluster_centers',
                        help="path for cluster_centers", type=str)

    parser.add_argument('--fpath_dist_src_tgt_index', default='Selection/src_tgt_model/cluster/dist_src_tgt_index_sort',
                        help="path for src_tgt distance", type=str)
    parser.add_argument('--fpath_dist_src_tgt_sort', default='Selection/src_tgt_model/cluster/dist_src_tgt_sort',
                        help="path for sorted src_tgt distance", type=str)

    parser.add_argument('--fpath_dist_src_tgt', default='Selection/src_tgt_model/cluster/dist_src_tgt',
                        help="path for sorted src_tgt distance", type=str)

    # ==-------------------------------------------------------------------------------------------------------------===

    options = parser.parse_args()

    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ------Training----------------------------
    options.run_cluster = True
    options.read_dist = False
    train_cluster(options, save_fea=True)
    # ------Eval-------------------------------
    options.run_cluster = False
    options.read_dist = True
    eval_model(options)
    plot_fea(options)




















































