import os
import numpy as np

from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import csv


# plot utils
def plot_scatter(*args, **kwargs):
    # plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


def pca_and_tsne_new(ori_ts, art_ts, sel_ts=None, experiment_name='', pics_dir='',
                     label_ori='Observed', label_syn='Synthetic', label_selected_src=None,
                     color_alpha=0.2, use_legend=False, fontsize=18):
    len_data = min(len(ori_ts), 4000)  # no more than 4000 points

    ori_ts = ori_ts[:len_data, :, :]
    art_ts = art_ts[:len_data, :, :]
    len_ori, len_art = len(ori_ts), len(art_ts)

    subplots = [231, 232, 233, 234, 235, 236]

    # Plot PCA and t-SNE in TimeGAN style
    ori_ts = np.mean(ori_ts, axis=-1, keepdims=False)  # (len_data, seq_len)
    art_ts = np.mean(art_ts, axis=-1, keepdims=False)  # (len_data, seq_len)
    if sel_ts is not None:
        len_sel = len(sel_ts)
        sel_ts = np.mean(sel_ts, axis=-1, keepdims=False)
        tsne_ts = np.concatenate((ori_ts, art_ts, sel_ts), axis=0)  # (3 * len_data, seq_len)
    else:
        tsne_ts = np.concatenate((ori_ts, art_ts), axis=0)  # (2 * len_data, seq_len)



    plt.figure(figsize=(16, 12))

    # ==---------------------------------------------------------PCA------------------------------------------------== #
    pca = PCA(n_components=2)
    pca.fit(ori_ts)
    pca_ori, pca_art, pca_sel = None, None, None
    pca_ori = pca.transform(ori_ts)
    pca_art = pca.transform(art_ts)

    if sel_ts is not None:
        pca_sel = pca.transform(sel_ts)

    plt.subplot(121)
    plt.grid()
    plot_scatter(pca_ori[:, 0], pca_ori[:, 1], color='b', alpha=color_alpha, label=label_ori)
    plot_scatter(pca_art[:, 0], pca_art[:, 1], color='r', alpha=color_alpha, label=label_syn)
    if sel_ts is not None:
        plot_scatter(pca_sel[:, 0], pca_sel[:, 1], color='k', alpha=color_alpha, marker='s', label=label_selected_src)


    # legend_params = {'weight': 'bold', 'size': 16}
    # plt.legend(prop=legend_params)
    if use_legend:
        # legend_params = {'weight': 'bold', 'size': 16}
        legend_params = {'size': fontsize}
        # plt.legend(prop=legend_params)
        plt.legend(prop=legend_params, bbox_to_anchor=(0, 1.05, 2.2, 0.5), loc="lower left", ncol=5, mode="expand")

    else:
        legend_params = {'size': 18}
        plt.legend(prop=legend_params)

    plt.title(f'PCA plots for features averaged', fontsize=18)
    plt.xlabel('x-pca', fontsize=18)
    plt.ylabel('y-pca', fontsize=18)

    # =-------------------------------------------------------TSNE--------------------------------------------------== #
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(tsne_ts)

    plt.subplot(122)
    plt.grid()

    plot_scatter(tsne_results[:len_data, 0],
                 tsne_results[:len_data, 1],
                 color='b', alpha=0.1, label=label_ori)

    plot_scatter(tsne_results[len_data:, 0],
                 tsne_results[len_data:, 1],
                 color='r', alpha=0.1, label=label_syn)

    if sel_ts is not None:
        plot_scatter(tsne_results[(len_ori + len_art):, 0],
                     tsne_results[(len_ori + len_art):, 1],
                     color='k', alpha=color_alpha, marker='s', label=label_selected_src)

    legend_params = {'size': 18}
    plt.legend(prop=legend_params)
    plt.title(f't-SNE plots for features averaged', fontsize=fontsize)
    plt.xlabel('x-tsne', fontsize=fontsize)
    plt.ylabel('y-tsne', fontsize=fontsize)

    file_name = f'{experiment_name}_visualization.png'

    file_dir = os.path.join(pics_dir, file_name)
    print('writing to {}'.format(file_dir))
    plt.savefig(file_dir)


def pca_and_tsne(ori_ts, art_ts, sel_ts=None, experiment_name='', pics_dir='',
                 label_ori='Observed', label_syn='Synthetic'):
    len_data = min(len(ori_ts), 4000)  # no more than 4000 points

    ori_ts = ori_ts[:len_data, :, :]
    art_ts = art_ts[:len_data, :, :]
    subplots = [231, 232, 233, 234, 235, 236]

    # Plot PCA and t-SNE in TimeGAN style
    ori_ts = np.mean(ori_ts, axis=-1, keepdims=False)  # (len_data, seq_len)
    art_ts = np.mean(art_ts, axis=-1, keepdims=False)  # (len_data, seq_len)
    tsne_ts = np.concatenate((ori_ts, art_ts), axis=0)  # (2 * len_data, seq_len)

    plt.figure(figsize=(16, 12))

    pca = PCA(n_components=2)
    pca.fit(ori_ts)
    pca_ori = pca.transform(ori_ts)
    pca_art = pca.transform(art_ts)

    plt.subplot(121)
    plt.grid()
    plot_scatter(pca_ori[:, 0], pca_ori[:, 1], color='b', alpha=0.1, label=label_ori)
    plot_scatter(pca_art[:, 0], pca_art[:, 1], color='r', alpha=0.1, label=label_syn)

    # legend_params = {'weight': 'bold', 'size': 16}
    # plt.legend(prop=legend_params)
    legend_params = {'size': 18}
    plt.legend(prop=legend_params)

    plt.title(f'PCA plots for features averaged', fontsize=18)
    plt.xlabel('x-pca', fontsize=18)
    plt.ylabel('y-pca', fontsize=18)

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(tsne_ts)

    plt.subplot(122)
    plt.grid()

    plot_scatter(tsne_results[:len_data, 0],
                 tsne_results[:len_data, 1],
                 color='b', alpha=0.1, label=label_ori)

    plot_scatter(tsne_results[len_data:, 0],
                 tsne_results[len_data:, 1],
                 color='r', alpha=0.1, label=label_syn)

    legend_params = {'size': 18}
    plt.legend(prop=legend_params)
    plt.title(f't-SNE plots for features averaged', fontsize=18)
    plt.xlabel('x-tsne', fontsize=18)
    plt.ylabel('y-tsne', fontsize=18)

    file_name = f'{experiment_name}_visualization.png'

    file_dir = os.path.join(pics_dir, file_name)
    print('writing to {}'.format(file_dir))
    plt.savefig(file_dir)