import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class BetaVAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, encoder, decoder, num_slots, slot_dim, z_dim = 6, nc=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = 32
        self.nc = nc
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.weight_init()

    def forward(self, x):
        zh = self.encoder(x)
        mu = zh[:, :self.z_dim]
        logvar = zh[:, self.z_dim:]
        z = reparameterize(mu, logvar)
        # import pdb; pdb.set_trace()
        x_recon = self.decoder(z.reshape(z.shape[0], self.num_slots, self.slot_dim)).view(x.size())
        return z.reshape(z.shape[0], self.num_slots, self.slot_dim), x_recon, mu, logvar

    def weight_init(self):
        for block in self._modules:
            kaiming_init(block)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def reparameterize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    # import pdb; pdb.set_trace()
    return mu + std*eps