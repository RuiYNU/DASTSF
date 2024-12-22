import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from torch.utils.data.sampler import Sampler
from torch.nn.modules.loss import _Loss
from Selection.tri_distance import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftMarginTriplet(_Loss):
    __constants__ = ['reduction']
    '''
    inputs: x1, x2 (dist)
    loss(x,y) = max(0, -y*(x1-x2)+margin)  y=1: first input should be ranked higher
    '''
    def __init__(self, margin=0, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(SoftMarginTriplet, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, dist_ap, dist_an, softmargin):
        softmargin = softmargin.to(DEVICE)

        loss = F.relu(dist_ap-dist_an+softmargin*self.margin)
        if self.reduction == 'elementwise_mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class _TripletLoss(object):
    def __init__(self, margin=None):
        self.margin = margin

        if margin is not None:
            self.ranking_loss = SoftMarginTriplet(margin=self.margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an, softmargin=None):
        # dist_ap, dist_an: [N]
        y = torch.ones_like(dist_ap)
        if self.margin is not None:

            loss = self.ranking_loss(dist_ap, dist_an, softmargin)
            return loss
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class STripletLoss(object):
    def __init__(self, params, weight_temp=0.5):
        self.params = params
        self.tri_loss_obj = _TripletLoss(margin=params['margin'])
        self.weight_temp = weight_temp

    def compute_weights(self, D_tgt_src_sort, apn_pos):
        a_pos, p_pos, n_pos = apn_pos[0], apn_pos[2], apn_pos[4]
        p_ind, n_ind = apn_pos[1], apn_pos[3]

        D_tgt_src_sort_n_neighbors = D_tgt_src_sort[:, :self.params["n_neighbors"]*2]

        weight = torch.ones(len(a_pos))
        for i in range(len(a_pos)):
            p_i_sort = D_tgt_src_sort_n_neighbors[a_pos[i]]
            weight[i] = (p_i_sort[n_ind[i]] - p_i_sort[p_ind[i]]) / (p_i_sort[-1] - p_i_sort[0])

        return weight

    def calculate(self, a_feat, p_feat, n_feat, apn_pos, D_tgt_src_sort):
        # a_feat: B * 1 * hidden-dim

        dist_ap = compute_correspond_dist(a_feat, p_feat, self.params['dist_type'])
        dist_an = compute_correspond_dist(a_feat, n_feat, self.params['dist_type'])

        softmargin = self.compute_weights(D_tgt_src_sort, apn_pos)

        loss = self.tri_loss_obj(dist_ap, dist_an, softmargin)

        # print('dist_ap:', dist_ap, 'dist_an:', dist_an, 'dist_n_max:', dist_max_an)
        return loss

    def __call__(self, a_pred, p_pred, n_pred, apn_pos, D_tgt_src_sort):
        loss = 0
        for i in range(len(a_pred)):
            res = self.calculate(a_pred[i], p_pred[i], n_pred[i],
                                 apn_pos, D_tgt_src_sort)
            loss += res

        if self.params['triloss_mean']:
            loss = loss / len(a_pred)
        return {'loss': loss}

class JointLoss(nn.Module):
    def __init__(self, margin):
        super(JointLoss, self).__init__()
        self.margin = margin
        self.sim_margin = 1 - margin/2

    def forward(self, features, agents, labels, similarity, features_target, similarity_target):

        loss_terms = []
        agents_arange = torch.arange(len(agents)).cuda()
        zero = torch.Tensor([0]).cuda()
        for (f, l, s) in zip(features, labels, similarity):
            loss_pos = (f - agents[l]).pow(2).sum()
            loss_terms.append(loss_pos)
            neg_idx = agents_arange != l
            hard_agent_idx = neg_idx & (s > self.sim_margin)

            if torch.any(hard_agent_idx):
                hard_neg_sdist = (f - agents[hard_agent_idx]).pow(2).sum(dim=1)
                loss_neg = torch.max(zero, self.margin - hard_neg_sdist).mean()
                loss_terms.append(loss_neg)

        for (f, s) in zip(features_target, similarity_target):
            # similarity_target: B * n_class
            hard_agent_idx = s > self.sim_margin
            if torch.any(hard_agent_idx):
                hard_neg_sdist = (f - agents[hard_agent_idx]).pow(2).sum(dim=1)
                loss_neg = torch.max(zero, self.margin - hard_neg_sdist).mean()
                loss_terms.append(loss_neg)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total


class DiscriminativeLoss(nn.Module):
    def __init__(self, mining_ratio=0.01):
        super(DiscriminativeLoss, self).__init__()
        self.mining_ratio = mining_ratio
        self.register_buffer('n_pos_pairs', torch.Tensor([0]))
        self.register_buffer('rate_TP', torch.Tensor([0]))
        self.moment = 0.1
        self.initialized = False

    def init_threshold(self, pairwise_agreements):
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = sorted_agreements[-pos]
        self.register_buffer('threshold', torch.Tensor([t]).cuda())
        self.initialized = True

    def forward(self, features, multilabels, labels):

        P, N = self._partition_sets(features.detach(), multilabels, labels)
        if P is None:
            pos_exponant = torch.Tensor([1]).cuda()
            num = 0
        else:
            sdist_pos_pairs = []
            for (i, j) in zip(P[0], P[1]):
                sdist_pos_pair = (features[i] - features[j]).pow(2).sum()
                sdist_pos_pairs.append(sdist_pos_pair)
            pos_exponant = torch.exp(-torch.stack(sdist_pos_pairs)).mean()
            num = -torch.log(pos_exponant)
        if N is None:
            neg_exponant = torch.Tensor([0.5]).cuda()
        else:
            sdist_neg_pairs = []
            for (i, j) in zip(N[0], N[1]):
                sdist_neg_pair = (features[i] - features[j]).pow(2).sum()
                sdist_neg_pairs.append(sdist_neg_pair)
            neg_exponant = torch.exp(- torch.stack(sdist_neg_pairs)).mean()
        den = torch.log(pos_exponant + neg_exponant)
        loss = num + den
        return loss

    def pair_idx_to_dist_idx(self, d, i, j):
        assert np.sum(i < j) == len(i)
        index = d * i - i * (i + 1) / 2 + j - 1 - i
        return index.astype(int)

    def dist_idx_to_pair_idx(self, d, i):
        if i.size == 0:
            return None
        b = 1 - 2 * d
        x = np.floor((-b - np.sqrt(b ** 2 - 8 * i)) / 2).astype(int)
        y = (i + x * (b + x + 2) / 2 + 1).astype(int)
        return x, y

    def _partition_sets(self, features, multilabels, labels):

        f_np = features.cpu().numpy()
        ml_np = multilabels.cpu().numpy()

        # labels_expand = labels.expand(len(labels), len(labels)) # B * B
        labels_expand = labels.reshape((labels.shape[0], -1))

        labels_expand_detach = labels_expand.detach().cpu()
        labels_agree = pdist(labels_expand_detach.cpu().numpy(), 'minkowski', p=1)
        idx = (labels_agree < 1e-6) # [labels_agree_num, ], [True, True, False, ...]

        similar_idx = np.array(range(len(labels_agree)))[idx]
        p_agree = 1 - pdist(ml_np, 'minkowski', p=1) / 2
        is_positive = p_agree[similar_idx] > self.threshold.item()

        pos_idx = similar_idx[is_positive] # similar_idx: labels_agree_num,
        neg_idx = similar_idx[~is_positive]

        P = self.dist_idx_to_pair_idx(len(f_np), pos_idx)
        N = self.dist_idx_to_pair_idx(len(f_np), neg_idx)
        # self._update_threshold(p_agree)
        self._update_threshold(p_agree[similar_idx])
        self._update_buffers(P, labels)
        return P, N

    def _update_threshold(self, pairwise_agreements):
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = torch.Tensor([sorted_agreements[-pos]]).cuda()
        self.threshold = self.threshold * (1 - self.moment) + t * self.moment

    def _update_buffers(self, P, labels):
        if P is None:
            self.n_pos_pairs = 0.9 * self.n_pos_pairs
            return 0
        n_pos_pairs = len(P[0])
        count = 0
        for (i, j) in zip(P[0], P[1]):
            count += labels[i] == labels[j]
        rate_TP = float(count) / n_pos_pairs
        self.n_pos_pairs = 0.9 * self.n_pos_pairs + 0.1 * n_pos_pairs
        self.rate_TP = 0.9 * self.rate_TP + 0.1 * rate_TP


class AnchorSampler(Sampler):
    def __init__(self, data_source, data_target, labels, k=10, index_sort=None):
        super().__init__(data_target)
        self.data_source = data_source
        self.data_target = data_target
        self.labels = labels
        self.index_sort = index_sort

        self.num_samples = len(data_target)
        if k*2 >= len(data_source):
            k = int(len(data_source) * 0.5)
        self.k = k

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        indices = torch.arange(self.num_samples)
        ret = []
        for i in indices:
            index = self.index_sort[i]

            pos_ind = int(np.random.random() * self.k)
            pos_pos = index[pos_ind] # k
            # ==---------------------------------------------------------==#
            neg_ind = int(np.random.random() * self.k) + self.k
            neg_pos = index[neg_ind]

            ret.append([i, pos_ind, pos_pos, neg_ind, neg_pos])

        return iter(ret)


class AnchorPreprocessor(object):
    def __init__(self, target_dataset, source_dataset, target_labels):
        super(AnchorPreprocessor, self).__init__()
        self.target_dataset = target_dataset
        self.source_dataset = source_dataset
        self.target_labels = target_labels
        # self.mask = mask

    def __len__(self):
        return len(self.target_dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            if len(indices) == 5:
                return self._get_triplet_item(indices)
            else:
                return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        return self.target_dataset[index], index

    def _get_triplet_item(self, index):
        a_feat = self.target_dataset[index[0]] # B * 1
        p_feat = self.source_dataset[index[2]] # n_pos * fea_dim
        n_feat = self.source_dataset[index[4]]
        position = index[0:]
        a_label = self.target_labels[index[0]]

        return a_feat, a_label, p_feat, n_feat, position

