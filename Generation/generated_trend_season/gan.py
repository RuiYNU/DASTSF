import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, seq_len):
        super(Generator, self).__init__()
        def block(inp, out):
            return nn.Sequential(
                nn.Linear(inp, out),
                nn.LayerNorm(out),
                nn.LeakyReLU(0.2),
            )

        self.block_0 = block(input_dim, input_dim)
        self.block_1 = block(input_dim, input_dim)
        self.block_2 = block(input_dim, hidden_dim)
        self.block_2_1 = block(hidden_dim, hidden_dim)
        self.block_3 = block(input_dim, hidden_dim*layers)
        self.block_3_1 = nn.Linear(hidden_dim*layers, hidden_dim*layers)
        self.final = nn.LeakyReLU(0.2)


        # trend_reshape: B * (seq_len*hidden_dim//2)
        self.half_hidden_dim = hidden_dim // 2
        self.block_4 = block(input_dim, seq_len*self.half_hidden_dim)
        self.block_4_1 = nn.Linear(seq_len*self.half_hidden_dim, seq_len*self.half_hidden_dim)

        self.block_5 = block(input_dim, seq_len*self.half_hidden_dim)
        self.block_5_1 = nn.Linear(seq_len*self.half_hidden_dim, seq_len*self.half_hidden_dim)

    def forward(self, x):
        # x: B * embed_dim (hidden_dim + layers*hidden_dim)
        x = self.block_0(x) + x
        x = self.block_1(x) + x
        x1 = self.block_2(x)
        x1 = self.block_2_1(x1) # x1: B * hidden_dim

        x2 = self.block_3(x) # x2: B * (hidden_dim * layers)
        x2 = self.block_3_1(x2) # x2: B * (hidden_dim*layers)

        trend = self.block_4(x)
        trend = self.block_4_1(trend)
        season = self.block_5(x)
        season = self.block_5_1(season)

        return torch.cat([x1, x2, trend, season], dim=-1)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, (2 * input_dim) // 3),
            nn.LeakyReLU(0.2),
            nn.Linear((2 * input_dim) // 3, input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 3, 1),
        )

    def forward(self, x):
        return self.model(x)