import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class InfoVAE(nn.Module):
    def __init__(self, nwave=1000, ncode=5, alpha=0, lambd=10000, 
                 nkernels=[3, 3, 3], nhiddens_enc=[128, 64, 32], nhiddens_dec=[128, 64, 32], npools=[2, 2, 2], 
                 dropout=0.2):
        super(InfoVAE, self).__init__()
        
        self.ncode = int(ncode)
        self.alpha = float(alpha)
        self.lambd = float(lambd)

        nkernel0, nkernel1, nkernel2 = nkernels
        nhidden0_enc, nhidden1_enc, nhidden2_enc = nhiddens_enc
        nhidden0_dec, nhidden1_dec, nhidden2_dec = nhiddens_enc
        npool0, npool1, npool2 = npools

        # convolutional layers
        self.conv0 = nn.Conv1d(1, 1, kernel_size=nkernel0)
        Lout = nwave - nkernel0 + 1 
        self.p0 = nn.MaxPool1d(npool0)
        Lout = int((Lout - npool0)/npool0 + 1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=nkernel1)
        Lout = Lout - nkernel1 + 1 
        self.p1 = nn.MaxPool1d(npool1)
        Lout = int((Lout - npool1)/npool1 + 1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=nkernel2)
        Lout = Lout - nkernel2 + 1 
        self.p2 = nn.MaxPool1d(npool2)
        Lout = int((Lout - npool2)/npool2 + 1)

        # encoders
        self.enc0 = nn.Linear(Lout, nhidden0_enc)
        self.d1 = nn.Dropout(p=dropout)
        self.enc1 = nn.Linear(nhidden0_enc, nhidden1_enc)
        self.d2 = nn.Dropout(p=dropout)
        self.enc2 = nn.Linear(nhidden1_enc, nhidden2_enc)
        self.d3 = nn.Dropout(p=dropout)

        self.mu = nn.Linear(nhidden2_enc, ncode)
        self.lv = nn.Linear(nhidden2_enc, ncode)
        
        # decoders
        self.decd = nn.Linear(ncode, nhidden2_dec)
        self.d3 = nn.Dropout(p=dropout)
        self.decd2 = nn.Linear(nhidden2_dec, nhidden1_dec)
        self.d4 = nn.Dropout(p=dropout)
        self.decd3 = nn.Linear(nhidden1_dec, nhidden0_dec)
        self.d5 = nn.Dropout(p=dropout)
        self.outp = nn.Linear(nhidden0_dec, nwave)
        
    def encode(self, x):
        x = self.p0(F.relu(self.conv0(x)))
        x = self.p1(F.relu(self.conv1(x)))
        x = self.p2(F.relu(self.conv2(x)))
        x = self.d1(F.leaky_relu(self.enc0(x)))
        x = self.d2(F.leaky_relu(self.enc1(x)))
        x = self.d3(F.leaky_relu(self.enc2(x)))

        mu = self.mu(x)
        logvar = self.lv(x)
        return mu[:,0,:], logvar[:,0,:]
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, x):
        x = self.d3(F.leaky_relu(self.decd(x)))
        x = self.d4(F.leaky_relu(self.decd2(x)))
        x = self.d5(F.leaky_relu(self.decd3(x)))
        x = self.outp(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    # https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        # The example code divides by (dim) here, making <kernel_input> ~ 1/dim
        # excluding (dim) makes <kernel_input> ~ 1
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)#/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)
    
    # https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    def compute_mmd(self, x, y):
        xx_kernel = self.compute_kernel(x,x)
        yy_kernel = self.compute_kernel(y,y)
        xy_kernel = self.compute_kernel(x,y)
        return torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2*torch.mean(xy_kernel)
    
    def loss(self, x, mask):
        recon_x, mu, logvar = self.forward(x)

        MSE = torch.sum(0.5 * ((x[:,0,:] - recon_x)[~mask]).pow(2))
        
        # KL divergence (Kingma and Welling, https://arxiv.org/abs/1312.6114, Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #return MSE + self.beta*KLD, MSE
                
        # https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        device = x.device
        true_samples = Variable(torch.randn(200, self.ncode), requires_grad=False)
        true_samples = true_samples.to(device)

        z = self.reparameterize(mu, logvar) #duplicate call
        z = z.to(device)
        # compute MMD ~ 1, so upweight to match KLD which is ~ n_batch x n_code
        MMD = self.compute_mmd(true_samples,z) * x.size(0) * self.ncode
        return MSE + (1-self.alpha)*KLD + (self.lambd+self.alpha-1)*MMD, MSE, KLD, MMD
