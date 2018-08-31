import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch


class CiDAN(nn.Module):
    def __init__(self, nodes_num_A, nodes_num_B, hidden_size, ZDIMS, batch_size):
        super(CiDAN, self).__init__()

        self.nodes_num_A = nodes_num_A
        self.nodes_num_B = nodes_num_B
        self.hidden_size = hidden_size
        self.ZDIMS = ZDIMS
        self.batch_size = batch_size

        self.relu = nn.ReLU()
        # encoder A
        self.encoder_fc_A = nn.Linear(nodes_num_A ** 2, hidden_size)
        self.encoder_mean_A = nn.Linear(hidden_size, ZDIMS)
        self.encoder_std_A = nn.Linear(hidden_size, ZDIMS)

        # encoder B
        self.encoder_fc_B = nn.Linear(nodes_num_B ** 2, hidden_size)
        self.encoder_mean_B = nn.Linear(hidden_size, ZDIMS)
        self.encoder_std_B = nn.Linear(hidden_size, ZDIMS)

        self.sigmoid = nn.Sigmoid()
        # decoder A
        self.decode_fc1_A = nn.Linear(ZDIMS * 2, hidden_size)
        self.decode_fc2_A = nn.Linear(hidden_size, nodes_num_A ** 2)

        # decoder B
        self.decode_fc1_B = nn.Linear(ZDIMS * 2, hidden_size)
        self.decode_fc2_B = nn.Linear(hidden_size, nodes_num_B ** 2)

    def encode(self, adjA, adjB):
        xA = adjA.view((self.batch_size, -1))
        xB = adjB.view((self.batch_size, -1))

        # encode graph A
        hA = self.relu(self.encoder_fc_A(xA))
        muA =  self.encoder_mean_A(hA)
        logvarA = self.encoder_std_A(hA)

        # encode graph B
        hB = self.relu(self.encoder_fc_B(xB))
        muB = self.encoder_mean_B(hB)
        logvarB = self.encoder_std_B(hB)

        return muA, logvarA, muB, logvarB

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, zA, zB):
        hA = self.relu(self.decode_fc1_A(torch.cat((zA, zB), 1)))
        outA = self.sigmoid(self.decode_fc2_A(hA))

        hB = self.relu(self.decode_fc1_B(torch.cat((zA, zB), 1)))
        outB = self.sigmoid(self.decode_fc2_B(hB))

        return outA, outB

    def forward(self, xA, xB):
        muA, logvarA, muB, logvarB = self.encode(xA, xB)

        zA = self.reparameterize(muA, logvarA)
        zB = self.reparameterize(muB, logvarB)

        zA, zB = self.decode(zA, zB)

        return zA, zB, muA, logvarA, muB, logvarB

    def loss_function(self, recon_xA, next_xA, muA, logvarA, recon_xB, next_xB, muB, logvarB):
        A_loss = self._single_loss_function(recon_xA, next_xA, muA, logvarA)
        B_loss = self._single_loss_function(recon_xB, next_xB, muB, logvarB)
        return A_loss + B_loss

    def _single_loss_function(self, recon_x, next_x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, next_x.view(self.batch_size, -1))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.batch_size * (self.nodes_num ** 2)

        return BCE + KLD


class VAE(nn.Module):
    def __init__(self, nodes_num, hidden_size, ZDIMS, batch_size):
        super(VAE, self).__init__()

        self.nodes_num = nodes_num
        self.hidden_size = hidden_size
        self.ZDIMS = ZDIMS
        self.batch_size = batch_size

        # encoder
        self.encoder_fc = nn.Linear(nodes_num ** 2, hidden_size)
        self.relu = nn.ReLU()
        self.encoder_mean = nn.Linear(hidden_size, ZDIMS)
        self.encoder_std = nn.Linear(hidden_size, ZDIMS)

        # decoder
        self.decode_fc1 = nn.Linear(ZDIMS, hidden_size)
        self.decode_fc2 = nn.Linear(hidden_size, nodes_num ** 2)
        self.sigmoid = nn.Sigmoid()

    def encode(self, adj):
        x = adj.view((self.batch_size, -1))

        # encode graph
        h1 = self.relu(self.encoder_fc(x))
        return self.encoder_mean(h1), self.encoder_std(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.decode_fc1(z))
        return self.sigmoid(self.decode_fc2(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, next_x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, next_x.view(self.batch_size, -1))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.batch_size * (self.nodes_num ** 2)

        return BCE + KLD
