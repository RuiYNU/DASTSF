import torch
import argparse
import pickle
from sys import maxsize
import sys
sys.path.append(".")
from Generation.generated_trend_season.train_aegan import AeGAN
from Generation.generated_trend_season.process_data import *
from Generation.generated_trend_season.visualization import *


def re_scale(data, scale_dict):
    # re_transform generated data according to train_scale_dict (min_max_scale)
    data_new = []
    # data: N * seq_len
    for i in range(data.shape[1]):

        temp = data[:, i, :]
        min_, scale_ = scale_dict['min'][i], scale_dict['max'][i]
        temp -= min_
        temp /= scale_
        if i == 0:
            data_new = temp
        else:
            data_new = np.concatenate([data_new, temp], axis=1)
    return data_new


def generated_tgt(syn, re_mask_data, seq_len, train_scale_dict, root_dir, fpath_result, pics_dir, experiment_name):
    generated_result = syn.synthesize(len(re_mask_data), seq_len=seq_len)
    re_generated_result = re_scale(np.array(generated_result), train_scale_dict)
    with open("{}/data".format(root_dir), "wb") as f:
        pickle.dump(generated_result, f)

    make_sure_path_exists(fpath_result)
    make_sure_path_exists(pics_dir)

    re_generated_result = re_generated_result.reshape(re_generated_result.shape[0], -1)
    save_csv(fpath_result, re_generated_result)
    re_generated_result = read_csv(fpath_result)

    if len(re_mask_data.shape)<3:
        re_mask_data = re_mask_data[:, :, np.newaxis]
        re_generated_result = re_generated_result[:, :, np.newaxis]

    pca_and_tsne(ori_ts=np.array(re_mask_data),
                 art_ts=np.array(re_generated_result), experiment_name=experiment_name,
                 pics_dir=pics_dir)


def read_syn_dataset(options):
    dataset = pickle.load(open(options.dataset, "rb"))
    train_set = dataset["train_set"]
    dyn_processor = dataset["processor"]
    dyn_processor.dim = options.dim
    train_set.set_input("dyn", "seq_len")

    train_scale_dict = dataset["train_scale_dict"]
    mask_scale_dict = dataset["mask_scale_dict"]
    normalize_target_train_data = dataset["normalize_train_data"]  # normalized_train_data
    normalize_target_mask_data = dataset["normalize_mask_data"]  # normalized_mask_data

    target_train_data = dataset["train_data"]  # original train_data
    target_mask_data = dataset["mask_data"]  # original mask data
    print('normalize_target_train_data:', normalize_target_train_data.shape,
          'normalize_target_mask_data:', normalize_target_mask_data.shape)

    return dyn_processor, train_set, train_scale_dict, mask_scale_dict, \
           normalize_target_train_data, normalize_target_mask_data, \
           target_train_data, target_mask_data


if __name__ == '__main__':
    # ===-----------------------------------------------------------------------===
    # Argument parsing
    # ===-----------------------------------------------------------------------===
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/tourism_yearly.pkl", dest="dataset", help=".pkl file to use")
    parser.add_argument("--device", default="0", dest="device", help="gpu")
    parser.add_argument("--task_name", dest="task_name", help="Name for this task")
    parser.add_argument("--root_dir", default="./tourism_result/", dest="root_dir",
                        help="Directory for models")
    parser.add_argument("--ae-epochs", default=2000, dest="ae_epochs", type=int,
                        help="training epochs for autoencoder")
    parser.add_argument("--gan-epochs", default=6000, dest="gan_epochs", type=int,
                        help="training epochs for WGAN")
    parser.add_argument("--d-update", default=5, dest="d_update", type=int,
                        help="discriminator updates per generator update")

    parser.add_argument("--eval-ae", dest="eval_ae", default=False, help="evaluate autoencoder")
    parser.add_argument("--seq-len", default=12, dest="seq_len", type=int, help="sequence_length")
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

    parser.add_argument("--fpath_result", default="Generation/generated_trend_season/result/tourism_yearly/re_scale_result_new", help="output result")
    parser.add_argument("--pics_dir", default="Generation/generated_trend_season//result/tourism_yearly/pics", help="output picture")

    options = parser.parse_args()
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # parameter set
    # options.dim = 1
    # options.seq_len = 8
    # options.ae_epochs = 1500
    # options.gan_epochs = 3000
    # options.task_name = "tourism_yearly"
    # options.root_dir = "Generation/generated_trend_season/result"
    # options.dataset = 'Generation/generated_trend_season/data/tourism_yearly.pkl'
    # options.training_ae = True
    # options.training_ae = False
    params = vars(options)

    # ==---------------Read Dataset----------------------------------------------------==
    dyn_processor, train_set, train_scale_dict, mask_scale_dict, \
    normalize_target_train_data, normalize_target_mask_data, \
    target_train_data, target_mask_data = read_syn_dataset(options=options)

    # ==---------------Read Dataset----------------------------------------------------==

    # ==---------------Train Model-----------------------------------------------------==
    root_dir = "{}/{}".format(options.root_dir, options.task_name)
    params["root_dir"] = root_dir
    params["seq_len"] = options.seq_len
    params["device"] = options.device
    re_mask_data = re_scale(normalize_target_mask_data, mask_scale_dict)

    syn = AeGAN(processors=dyn_processor, params=params)

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


    # fpath_result = 'Generation/generated_trend_season/result/tourism_yearly/re_scale_result_new'
    #     # pics_dir = 'Generation/generated_trend_season//result/tourism_yearly/pics'
    fpath_result = options.fpath_result
    pics_dir = options.pics_dir
    generated_tgt(syn=syn, re_mask_data=re_mask_data, seq_len=params["seq_len"], train_scale_dict=train_scale_dict,
                  root_dir=root_dir, fpath_result=fpath_result, pics_dir=pics_dir, experiment_name=options.task_name)

















