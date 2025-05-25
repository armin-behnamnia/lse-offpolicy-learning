import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def conv_bn_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
    )
def tconv_bn_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
  return nn.Sequential(
      nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
      nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
  )
def tconv_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
  return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

def conv_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

def fc_layer(in_features,out_features):
  return nn.Linear(in_features,out_features)

def fc_bn_layer(in_features,out_features):
  return nn.Sequential(
      nn.Linear(in_features,out_features),
      nn.BatchNorm1d(out_features)
  )

latent_dim =100

gf_dim = 64
df_dim = 64

in_h = 64
in_w = 64
c_dim = 1

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))
s_h, s_w = in_h, in_w
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

class Q(nn.Module):
  def __init__(self):
    super(Q,self).__init__()
    self.projection_z = fc_bn_layer(latent_dim,s_h16*s_w16*gf_dim*8)

    self.theta_params = nn.Sequential(
    tconv_bn_layer(gf_dim*8,gf_dim*4,4,stride=2,padding=1),
    nn.ReLU(),
    tconv_bn_layer(gf_dim*4,gf_dim*2,4,stride=2,padding=1),
    nn.ReLU(),
    tconv_bn_layer(gf_dim*2,gf_dim,4,stride=2,padding=1),
    nn.ReLU(),
    tconv_layer(gf_dim,c_dim,4,stride=2,padding=1),
    nn.Tanh()
    )
    
  def forward(self, x):
    x = F.relu(self.projection_z(x).view(-1,gf_dim*8,s_h16,s_w16))
    x =  self.theta_params(x)
    return x

class V(nn.Module):
  def __init__(self):
    super(V,self).__init__()
    self.w_params = nn.Sequential (
        conv_layer(c_dim,df_dim,4,stride=2,padding=1),
        nn.LeakyReLU(0.1),
        conv_bn_layer(df_dim,df_dim*2,4,stride=2,padding=1),
        nn.LeakyReLU(0.1),
        conv_bn_layer(df_dim*2,df_dim*4,4,stride=2,padding=1),
        nn.LeakyReLU(0.1),
        conv_bn_layer(df_dim*4,df_dim*8,4,stride=2,padding=1),
        nn.LeakyReLU(0.1),
        nn.Flatten(1),
        fc_layer(df_dim*8*s_h16*s_w16,1)
    )

  def forward(self, x):
    x = self.w_params(x)
    return x

def positive_elu(x):
  return (x + 1) * (x > 0).detach().float() + torch.exp(x) * (x < 0).detach().float()

class Activation_g(nn.Module):
  def __init__(self,divergence="GAN"):
    super(Activation_g,self).__init__()
    self.divergence =divergence
  def forward(self,v):
    divergence = self.divergence
    if divergence == "KLD":
      return v
    elif divergence == "RKL":
      return -(F.elu(-v) + 1) #-torch.exp(-v)
    elif divergence == "CHI":
      return v
    elif divergence == "SQH":
      return 1-torch.exp(-v)
    elif divergence == "JSD":
      return torch.log(torch.tensor(2.))-torch.log(1.0+torch.exp(-v) + 1e-8)
    elif divergence == "GAN":
      return -torch.log(1.0+torch.exp(-v) + 1e-8) # log sigmoid

class Conjugate_f(nn.Module):
  def __init__(self,divergence="GAN"):
    super(Conjugate_f,self).__init__()
    self.divergence = divergence
  def forward(self,t):
    divergence= self.divergence
    if divergence == "KLD":
      return torch.exp(t-1)
    elif divergence == "RKL":
      return -1 -torch.log(-t + 1e-8)
    elif divergence == "CHI":
      return 0.25*t**2+t
    elif divergence == "SQH":
      return t/(torch.tensor(1.)-t)
    elif divergence == "JSD":
      return -torch.log(2.0-torch.exp(t) + 1e-8)
    elif divergence == "GAN":
      return  -torch.log(1.0-torch.exp(t) + 1e-8)

class VLOSS(nn.Module):
  def __init__(self,divergence="GAN"):
    super(VLOSS,self).__init__()
    self.conjugate = Conjugate_f(divergence)
    self.activation = Activation_g(divergence)
  def forward(self, v1, v2):
    return torch.mean(self.activation(v1)) - torch.mean(self.conjugate(self.activation(v2)))
  

# class QLOSS(nn.Module):
#   def __init__(self,divergence="GAN"):
#     super(QLOSS,self).__init__()
#     self.conjugate = Conjugate_f(divergence)
#     self.activation = Activation_g(divergence)
#   def forward(self,v):
#     return torch.mean(-self.conjugate(self.activation(v)))
  

class QIPSLOSS(nn.Module):
  def __init__(self,divergence="GAN"):
    super(QIPSLOSS,self).__init__()
    self.conjugate = Conjugate_f(divergence)
    self.activation = Activation_g(divergence)
  def forward(self, v, prop, output):
    u = self.activation(v)
    w = self.conjugate(u)
    # print("QIPS LOSS =", v, u, w)
    return torch.mean(w * output / prop)

class QIPSLOSS2(nn.Module):
  def __init__(self,divergence="GAN"):
    super(QIPSLOSS2,self).__init__()
    self.conjugate = Conjugate_f(divergence)
    self.activation = Activation_g(divergence)
  def forward(self, v, prop, output):
    u = self.activation(v)
    # print("QIPS LOSS =", v, u, w)
    return torch.mean(u * output / prop)