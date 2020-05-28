# deep learning models
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_ as kaiming
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class Beta_VAE(nn.Module):
    def __init__(self, z_size=10,in_channel=3):
        super(Beta_VAE, self).__init__()
        self.z_size = z_size
        self.in_channel=in_channel

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channel, 32, 4, 2, 1),          
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            Reshape((-1, 512)),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, self.z_size*2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            Reshape((-1, 32, 4, 4)),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.in_channel, 4, 2, 1),
        )

        self.init_weights(self.encoder)
        self.init_weights(self.decoder)

    def init_weights(self,modules):
        for module in modules:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                kaiming(module.weight,nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
        pass

    def reparametrize(self, mu, logvar): # https://zhuanlan.zhihu.com/p/27549418
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            z = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            z = torch.FloatTensor(std.size()).normal_()
        return z*std+mu
        # return z.mul(std).add_(mu)
    
    def KL_divergence(self,mean,logvar):# TODO: why the kld make mean goes to zero and logvar goes to -100?
        kl_divergence = mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kl_divergence = (kl_divergence).mul_(-0.5)
        kl_divergence_each_sample=torch.sum(kl_divergence.sum(1))/mean.shape[0]
        kl_divergence_each_dim=kl_divergence.mean(0)
        return kl_divergence_each_sample, kl_divergence_each_dim
        # kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())/mean.shape[0]
        # return kl_divergence
        pass


    def forward(self, x):
        mean_var = self.encoder(x)
        std_normal=torch.randn((x.shape[0],self.z_size))
        mean = mean_var[:, :self.z_size]
        logvar = mean_var[:, self.z_size:] # predict log varance because it is more stable and make sure var should always be positive https://stats.stackexchange.com/questions/353220/why-in-variational-auto-encoder-gaussian-variational-family-we-model-log-sig
        z = self.reparametrize(mean,logvar)
        output = self.decoder(z).view(x.size())

        return output, mean, logvar
    
    def forward_with_mean(self,x,loss="normal"):
        mean_var = self.encoder(x)
        mean = mean_var[:, :self.z_size]
        output = self.decoder(mean).view(x.size())
        if loss=="bernoulli":
            output=torch.sigmoid(output)

        return output, mean

    def forward_explore_var(self,x,axis=0,interval=0.3,max_range=3,loss="normal"):
        assert x.shape[0]==1
        var_range=torch.arange(-max_range, max_range+0.01, interval)
        num_image=len(var_range)
        mean_var = self.encoder(x)
        mean = mean_var[:, :self.z_size]
        means=mean.repeat(num_image,1)
        # print(means.shape)
        for index in range(num_image):
            means[index,axis]=var_range[index]
        output = self.decoder(means).view((len(var_range),x.shape[1],x.shape[2],x.shape[3]))
        if loss=="bernoulli":
            output=torch.sigmoid(output)
        return output

def get_model(args,use_cuda):
    if args.dataset=="Coloured dSprites":
        model=Beta_VAE()
        pass
    elif args.dataset=="3dchairs":
        model=Beta_VAE(10,3)
        pass
    elif args.dataset=="celeba":
        model=Beta_VAE(32,3)
        pass
    elif args.dataset=="dsprites":
        model=Beta_VAE(10,1)
        pass
    else:
        raise ValueError("data set not found! ")
    if use_cuda:
        return model.cuda()
    else:
        return model
