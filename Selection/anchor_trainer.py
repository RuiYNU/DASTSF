import torch
import os
from fastNLP import seq_len_to_mask
from Selection.anchor_loss import *
# from Selection.tri_loss_test import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeaEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout = dropout

        self.rnn = nn.GRU(input_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.final = nn.LeakyReLU(0.2)

    def forward(self, x):
        bs, max_len, _ = x.size()
        seqlen = torch.Tensor([max_len]*bs)
        #seqlen = max_len
        x = x.to(DEVICE)

        packed = nn.utils.rnn.pack_padded_sequence(x, seqlen, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        #==------------------------------------------==#
        seqlen = seqlen.to(DEVICE)
        #==------------------------------------------==#
        h1, _ = self.max_pooling(out, seqlen)
        h2 = self.mean_pooling(out, seqlen)
        h3 = h.view(self.layers, -1, bs, self.hidden_dim)[-1].view(bs, -1)

        # h3: B * (layers * hidden_dim)
        glob = torch.cat([h1, h2, h3], dim=-1)
        glob = self.final(self.fc(glob))
        h3 = h.permute(1, 0, 2).contiguous().view(bs, -1)
        # h3 = self.final(self.fc1(h3))
        hidden = torch.cat([glob, h3], dim=-1)
        return hidden

    def mean_pooling(self, tensor, seq_len, dim=1):
        mask = seq_len_to_mask(seq_len)
        mask = mask.view(mask.size(0), mask.size(1), -1).float()
        return torch.sum(tensor * mask, dim=dim) / seq_len.unsqueeze(-1).float()

    def max_pooling(self, tensor, seq_len, dim=1):
        mask = seq_len_to_mask(seq_len)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = mask.expand(-1, -1, tensor.size(2)).float()
        return torch.max(tensor + mask.le(0.5).float() * -1e9, dim=dim)

class FeaModel(nn.Module):
    def __init__(self, params, input_dims, fea_dims, hidden_dims, layers,
                 dropout=0.1, alpha=0.5, device=DEVICE):
        super().__init__()
        self.params = params
        self.num_class = params['cluster']

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.fea_dims = fea_dims
        self.device = device
        self.alpha = alpha

        self.encoder_q = FeaEncoder(input_dims, hidden_dims, layers, dropout)
        self.fc = nn.Linear(hidden_dims*4, self.num_class)

    def forward(self, x):
        #############################################
        x = x.float()
        #############################################

        feature = self.encoder_q(x) # B * hidden_dim
        feature = feature.renorm(2, 0, 1e-5).mul(1e5)
        w = self.fc.weight # num_class * hidden_dim
        ww = w.renorm(2, 0, 1e-5).mul(1e5)
        # print('features:', feature.shape, 'ww:', ww.shape)
        sim = feature.mm(ww.t()) # B * n_class
        return feature, sim


class Trainer(object):
    def __init__(self, params, input_dims, fea_dims, hidden_dims, layers,
                 N_tgt_samples, D_tgt_src_sort,
                 dropout=0.1, alpha=0.5, print_inter=5, device=DEVICE):
        super(Trainer, self).__init__()
        self.params = params
        # self.D_tgt_src = D_tgt_src
        self.D_tgt_src_sort = D_tgt_src_sort
        self.print_inter = print_inter

        self.al_loss = nn.CrossEntropyLoss().cuda()
        self.rj_loss = JointLoss(params['margin']).cuda()
        self.mdl_loss = DiscriminativeLoss(params['mining_ratio']).cuda()


        self.tri_loss = STripletLoss(params, weight_temp=0.5)

        self.net = FeaModel(params, input_dims, fea_dims, hidden_dims, layers,
                            dropout=dropout, alpha=alpha, device=device)
        fea_model_parameters = list(self.net.parameters())
        self.optimizer = torch.optim.Adam(fea_model_parameters, lr=0.001)

        self.multilabel_ini = torch.zeros(N_tgt_samples, params['cluster'])
        self.initilized = self.multilabel_ini.sum(dim=1) != 0
        self.net = self.net.to(DEVICE)

    def init_loss(self, target_loader):
        features, labels, sim = [], [], []
        for i, data in enumerate(target_loader):
            # x, y = data[0].cuda(), data[1]
            x, y, p_feat, n_feat, position = data[0].cuda(), data[1].cuda(), \
                                             data[2].cuda(), data[3].cuda(), \
                                             data[4]
            if len(x.shape) < 3:
                x = x[:, :, np.newaxis]

            with torch.no_grad():
                feature_batch, sim_batch = self.net(x)
            features.append(feature_batch)
            labels.append(y)
            sim.append(sim_batch)

        sim = torch.cat(sim, dim=0)
        multilables = F.softmax(sim, dim=1) # sim : B * n_class
        ml_np = multilables.cpu().numpy()
        pairwise_agreements = 1 - pdist(ml_np, 'minkowski', p=1) / 2
        self.mdl_loss.init_threshold(pairwise_agreements)

    def train_epoch(self, target_loader, epoch):
        self.init_loss(target_loader)
        self.net.train()

        root_dir = self.params["model_save_path"]

        for kk in range(epoch):
            tot, loss_total = 0, 0
            for i, target_tuple in enumerate(target_loader):
                data, labels, p_feat, n_feat, position = target_tuple[0].cuda(), target_tuple[1].cuda(),\
                                                         target_tuple[2].cuda(), target_tuple[3].cuda(), \
                                                         target_tuple[4]
                idx_target = position[0]

                if len(data.shape) < 3:
                    data = data[:, :, np.newaxis]
                    p_feat = p_feat[:, :, np.newaxis]
                    n_feat = n_feat[:, :, np.newaxis]

                features, similarity = self.net(data)
                scores = similarity
                # print('scores:', scores.shape, 'labels:', labels)
                labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)
                loss_target = self.al_loss(scores, labels)
                agents = self.net.fc.weight.renorm(2, 0, 1e-5).mul(1e5)

                with torch.no_grad():
                    agents_detach = agents.t_()

                multilabels = F.softmax(features.mm(agents_detach), dim=1)  # B * n_class
                multilabels = multilabels.to(DEVICE)

                #==-----------------------------------------------------------------------==#
                multilabels_cpu = multilabels.detach().cpu()
                is_init_batch = self.initilized[idx_target]
                initialized_idx = idx_target[is_init_batch]
                uninitialized_idx = idx_target[~is_init_batch]
                self.multilabel_ini[uninitialized_idx] = multilabels_cpu[~is_init_batch]
                self.initilized[uninitialized_idx] = 1
                self.multilabel_ini[initialized_idx] = 0.9 * self.multilabel_ini[initialized_idx] \
                                                          + 0.1 * multilabels_cpu[is_init_batch]
                loss_target_mdl = self.mdl_loss(features, self.multilabel_ini[idx_target], labels)


                data_copy = data
                a_t, _ = self.net(data_copy)
                p_t, _ = self.net(p_feat)
                n_t, _ = self.net(n_feat)
                if len(a_t.shape) < 3:
                    a_t = a_t[:, np.newaxis, :]
                    p_t = p_t[:, np.newaxis, :]
                    n_t = n_t[:, np.newaxis, :]
                loss_tri = self.tri_loss(a_t, p_t, n_t, position, self.D_tgt_src_sort)
                #==----------------------------------------------------------------------==#

                self.optimizer.zero_grad()
                # loss_target_mdl, loss_target = 0.0, 0.0
                loss_mdl_al = loss_target + loss_target_mdl
                loss_tri_ = loss_tri['loss']

                # loss_total = loss_tri_ + loss_mdl_al
                loss_total = loss_tri_ + 0.2 * loss_mdl_al
                loss_total.backward()
                self.optimizer.step()
                tot += 1

            if kk % self.print_inter == 0:
                loss_total = loss_total / tot
                print('Epoch:{}, loss_total:{}'.format(kk,
                                           loss_total.detach().cpu().numpy().mean().tolist(),
                                           ))

            if kk % 100 == 0:
                torch.save(self.net.state_dict(), '{}/fea_model_{}.dat'.format(root_dir, str(kk)))




