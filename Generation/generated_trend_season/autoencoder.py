# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.nn import functional as F
from fastNLP import seq_len_to_mask
import random
import numpy as np
import math
import torch.fft as fft
from einops import reduce, rearrange
# from Generation.generated_trend_season.test_trend_seasonal import EncoderLayer, DecoderLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_pooling(tensor, seq_len, dim=1):
    mask = seq_len_to_mask(seq_len)
    mask = mask.view(mask.size(0), mask.size(1), -1).float()
    return torch.sum(tensor * mask, dim=dim) / seq_len.unsqueeze(-1).float()


def max_pooling(tensor, seq_len, dim=1):
    mask = seq_len_to_mask(seq_len)
    mask = mask.view(mask.size(0), mask.size(1), -1)
    mask = mask.expand(-1, -1, tensor.size(2)).float()
    return torch.max(tensor + mask.le(0.5).float() * -1e9, dim=dim)


class BandFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()
        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class EncoderLayer(nn.Module):
    def __init__(self, input_dims, kernels, length):
        super().__init__()
        if input_dims > 1:
            component_dims = input_dims // 2
        else:
            component_dims = input_dims
        self.kernels = kernels
        self.length = length

        self.repr_dropout = nn.Dropout(p=0.1)

        self.tfd = nn.ModuleList([nn.Conv1d(input_dims, component_dims, k, padding=k - 1) for k in kernels])
        self.sfd = nn.ModuleList([BandFourierLayer(input_dims, component_dims, b, 1, length=length) for b in range(1)]
                                 )

    def forward(self, x):
        # x: B * seq_len * (layer_num*hidden_dim)
        x = x.transpose(1, 2)  # B * () * seq_len
        trend = []
        for idx, layer in enumerate(self.tfd):
            out = layer(x)
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # B * seq_len * ()

        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        seasons = []
        x = x.transpose(1, 2)  # B * seq_len * ()
        for layer in self.sfd:
            season = layer(x) # 448 * 12 * 1
            season = self.repr_dropout(season)
            seasons.append(season)
        seasons = sum(seasons)
        return trend, seasons


class DecoderLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_out = c_out
        self.pred = nn.Linear(c_in, c_out)

    def forward(self, trend, seasons):

        trend = self.pred(trend)
        season = self.pred(seasons)
        return trend, season


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, layers, dropout, seq_len=12):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.rnn = nn.GRU(input_dim, hidden_dim, layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.final = nn.LeakyReLU(0.2)

        self.encoder_layer = EncoderLayer(input_dims=hidden_dim, kernels=[1, 2, 4, 8, 16, 32, 64], length=seq_len)

    def forward(self, dynamics, seq_len):
        # dynamic: batch_size * seq_len * 1
        bs, max_len, _ = dynamics.size()
        x = dynamics

        packed = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # out: B * seq_len * hidden_dim ---- h: layers * B * hidden_dim
        trend, season = self.encoder_layer(out)

        # seq_len_ = seq_len
        seq_len = seq_len.to(DEVICE)

        trend_season = torch.cat([trend, season], dim=-1)  # trend_season: B * seq_len * hidden_dim
        h1, _ = max_pooling(trend_season, seq_len)
        h2 = mean_pooling(trend_season, seq_len)

        h3 = h.view(self.layers, -1, bs, self.hidden_dim)[-1].view(bs, -1)
        # h3: B * (layers * hidden_dim)
        glob = torch.cat([h1, h2, h3], dim=-1)
        glob = self.final(self.fc(glob))

        h3 = h.permute(1, 0, 2).contiguous().view(bs, -1)

        hidden = torch.cat([glob, h3], dim=-1)
        trend = trend.reshape(trend.shape[0], -1)
        season = season.reshape(season.shape[0], -1)
        hidden = torch.cat([hidden, trend, season], dim=-1)
        # glob: B*hidden_dim, h3: B*(layers*hidden_dim), trend+season: B*(seq_len *hidden_dim)

        return hidden


class Decoder(nn.Module):
    def __init__(self, processors, hidden_dim, dynamics_dim, layers, c_out, dropout):
        super(Decoder, self).__init__()
        # 1111self.s_P, self.d_P = processors

        self.d_P = processors
        self.hidden_dim = hidden_dim
        self.dynamics_dim = dynamics_dim
        self.layers = layers
        self.rnn = nn.GRU(hidden_dim + dynamics_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.dynamics_fc = nn.Linear(hidden_dim, dynamics_dim)

        self.decoder_layer = DecoderLayer(c_in=hidden_dim // 2, c_out=c_out)

    def forward(self, embed, dynamics, seq_len, forcing=0.5):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:(self.layers + 1) * self.hidden_dim]
        half_hidden_dim = self.hidden_dim // 2
        trend = embed[:, (self.layers + 1) * self.hidden_dim:-half_hidden_dim * seq_len]
        season = embed[:, -half_hidden_dim * seq_len:]
        trend = trend.reshape(trend.shape[0], seq_len, -1)
        season = season.reshape(season.shape[0], seq_len, -1)
        trend_, season_ = self.decoder_layer(trend, season)
        # glob: B * hidden_dim   hidden: B * (3*hidden_dim)

        glob = glob.unsqueeze(1)
        bs, max_len, _ = dynamics.size()
        x = dynamics[:, 0:1, :]
        hidden = hidden.view(bs, self.layers, -1).permute(1, 0, 2).contiguous()
        res = []
        for i in range(max_len):
            # glob: B * 1 * hidden_dim   x: B * 1 * dim
            x = torch.cat([glob, x.detach()], dim=-1)
            out, hidden = self.rnn(x, hidden)
            # out: B * 1 * dim
            out = apply_activation(self.d_P, self.dynamics_fc(out).squeeze(1)).unsqueeze(1)
            if random.random() > forcing:
                x = out
            else:
                x = dynamics[:, i + 1:i + 2, :]
            res.append(out)
        res = torch.cat(res, dim=1)
        return res + trend_ + season_

    def generate_dynamics(self, embed, max_len):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:(self.layers + 1) * self.hidden_dim]

        half_hidden_dim = self.hidden_dim // 2
        trend = embed[:, (self.layers + 1) * self.hidden_dim:-half_hidden_dim * max_len]
        season = embed[:, -max_len * half_hidden_dim:]

        trend = trend.reshape(trend.shape[0], max_len, -1)
        season = season.reshape(season.shape[0], max_len, -1)
        trend_, season_ = self.decoder_layer(trend, season)

        glob = glob.unsqueeze(1)
        bs = glob.size(0)
        x = torch.zeros((bs, 1, self.dynamics_dim)).to(embed.device)
        hidden = hidden.view(bs, self.layers, -1).permute(1, 0, 2).contiguous()
        res = []
        for i in range(max_len):
            x = torch.cat([glob, x], dim=-1)
            out, hidden = self.rnn(x, hidden)
            out = apply_activation(self.d_P, self.dynamics_fc(out).squeeze(1)).detach()
            x = out.unsqueeze(1)
            res.append(x)
        res = torch.cat(res, dim=1)
        res += trend_
        res += season_
        return res.cpu().numpy()


class Autoencoder(nn.Module):
    def __init__(self, processors, hidden_dim, embed_dim, layers, seq_len, dropout=0.0):
        super(Autoencoder, self).__init__()

        # 1111statics_dim, dynamics_dim = 0, processors[1].dim
        dynamics_dim = processors.dim
        self.encoder = Encoder(dynamics_dim, hidden_dim, embed_dim, layers, dropout, seq_len=seq_len)
        self.decoder = Decoder(processors, hidden_dim, dynamics_dim, layers,
                               c_out=dynamics_dim, dropout=dropout)

    def forward(self, dynamics, seq_len):
        hidden = self.encoder(dynamics, seq_len)
        input_x = torch.zeros_like(dynamics[:, 0:1, :])
        input_x = torch.cat([input_x, dynamics[:, :-1, :]], dim=1)
        return self.decoder(hidden, input_x, seq_len[0])


def apply_activation(processors, x):
    data = []
    st = 0
    for model in processors.models:
        ed = model.length + st
        if ed > x.shape[-1]:
            break
        data.append(torch.sigmoid(x[:, st:ed]))
        st = ed

    return torch.cat(data, dim=-1)


