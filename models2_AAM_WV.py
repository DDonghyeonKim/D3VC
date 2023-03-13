"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import os
import os.path as osp

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ECAPA_TDNN.model import ECAPA_TDNN
from conformer import Conformer
class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance
"""
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.Num_fet=num_features
        self.norm = nn.LayerNorm(num_features, affine=False)
        self.fc = nn.Sequential(
            nn.Linear(style_dim, num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features * 2),
            nn.SiLU()
                                )
        self.fc2 = nn.Sequential(
            nn.Linear(style_dim, num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features * 2),
            nn.SiLU()
                                )

        self.alpha1 = nn.Parameter(torch.FloatTensor(num_features))
        nn.init.constant_(self.alpha1, 1)
        self.alpha2 = nn.Parameter(torch.FloatTensor(num_features))
        nn.init.constant_(self.alpha2, 0.0)
        self.beta1 = nn.Parameter(torch.FloatTensor(num_features))
        nn.init.constant_(self.beta1, 0.0)
        self.beta2 = nn.Parameter(torch.FloatTensor(num_features))
        nn.init.constant_(self.beta2, 0.0)

        self.labmda_a = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.labmda_a, 1)
        self.labmda_b = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.labmda_b, 0.5)

    def forward(self, x, s):
        ##x => B,640,5,48
        ##s => B,192
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        h2 = self.fc2(s)
        h2 = h2.view(h.size(0), h2.size(1), 1, 1)
        gamma2, beta22 = torch.chunk(h2, chunks=2, dim=1)

        alpha1=torch.reshape(self.alpha1,[1,self.Num_fet,1,1])
        beta1=torch.reshape(self.beta1,[1,self.Num_fet,1,1])
        alpha2=torch.reshape(self.alpha2,[1,self.Num_fet,1,1])
        beta2=torch.reshape(self.beta2,[1,self.Num_fet,1,1])
        return torch.maximum((alpha1 + gamma*(self.labmda_a)) * self.norm(x+0.00001) + (self.labmda_b*beta+beta1),(alpha2 + gamma2*(self.labmda_a)) * self.norm(x+0.00001) + (self.labmda_b*beta22+beta2))
"""

"""

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.Num_fet=num_features
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Sequential(
            nn.Linear(style_dim, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features * 2),
            nn.SiLU()
                                )

        self.alpha1 = nn.Parameter(torch.FloatTensor(num_features))
        nn.init.constant_(self.alpha1, 1)
        self.alpha2 = nn.Parameter(torch.FloatTensor(num_features))
        nn.init.constant_(self.alpha2, 0.0)
        self.beta1 = nn.Parameter(torch.FloatTensor(num_features))
        nn.init.constant_(self.beta1, 0.0)
        self.beta2 = nn.Parameter(torch.FloatTensor(num_features))
        nn.init.constant_(self.beta2, 0.0)

        self.labmda_a = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.labmda_a, 1)
        self.labmda_b = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.labmda_b, 0.5)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)


        alpha1=torch.reshape(self.alpha1,[1,self.Num_fet,1,1])
        beta1=torch.reshape(self.beta1,[1,self.Num_fet,1,1])
        alpha2=torch.reshape(self.alpha2,[1,self.Num_fet,1,1])
        beta2=torch.reshape(self.beta2,[1,self.Num_fet,1,1])
        return torch.maximum((alpha1 + gamma*(self.labmda_a)) * self.norm(x) + (self.labmda_b*beta+beta1),(alpha2 + gamma*(self.labmda_a)) * self.norm(x) + (self.labmda_b*beta+beta2))
"""

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s):
        h = (self.fc(s))
        h = h.view(h.size(0), h.size(1), 1, 1)+0.00001
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample='none'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=48*8, w_hpf=1, F0_channel=0):
        super().__init__()

        self.stem = nn.Conv2d(1, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 1, 1, 1, 0))
        self.F0_channel = F0_channel
        self.upsample1=nn.ConvTranspose1d(384*5,768,kernel_size=5,stride=2, padding=2)
        self.upsample2=nn.ConvTranspose1d(768,768,kernel_size=5,stride=2, padding=2)
        #self.upsample2=nn.ConvTranspose1d(768,768,kernel_size=3,stride=2, padding=2)
        # down/up-sampling blocks
        repeat_num = 4 #int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1

        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = 'timepreserve'
            else:
                _downtype = 'half'

            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=_downtype))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=_downtype))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
        
        # F0 blocks 
        if F0_channel != 0:
            self.decode.insert(
                0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out, style_dim, w_hpf=w_hpf))
        
        # bottleneck blocks (decoder)
        for _ in range(2):
            self.decode.insert(
                    0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out + int(F0_channel / 2), style_dim, w_hpf=w_hpf))
        
        if F0_channel != 0:
            self.F0_conv = nn.Sequential(
                ResBlk(F0_channel, int(F0_channel / 2), normalize=True, downsample="half"),
            )
        

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def Middle(self, x):
        x = self.stem(x)
        cache = {}
        Feat=[]
        for block in self.encode:
            x = block(x)
        sizes=x.shape###-1,384, 5, 48
        x_re=torch.reshape(x,[sizes[0],384*5,sizes[3]])#119
        up1=self.upsample1(x_re)#-1, 768, 96
        up2=self.upsample2(up1)#-1, 768, 96
        return up2


    def forward(self, x, s, masks=None, F0=None):            
        x = self.stem(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        ##-1,
        #print(up2.shape)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])

        return self.to_out(x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=48, num_domains=60, hidden_dim=384):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, num_domains=2, max_conv_dim=384):
        super().__init__()

        self.speaker_encoder = ECAPA_TDNN(C=1024)
        #self.LN=nn.LayerNorm(192)
        #self.unshared = nn.ModuleList()
        #for _ in range(num_domains):
        #    self.unshared += [nn.Linear(192, style_dim)]

    def forward(self, x, y):
        s=self.speaker_encoder(x)
        return s





#VQVC
#256/ 0.3939
#64/ 0.31
#VQVC+
#512/ 0.04
#64/ 0.05

class Discriminator(nn.Module):
    def __init__(self, dim_in=48, num_domains=60, max_conv_dim=384, repeat_num=4):
        super().__init__()
        
        # real/fake discriminator
        self.dis1 = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        # adversarial classifier
        self.cls1 = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)


        self.num_domains = num_domains
        self.AAMsoftmax=AAMsoftmax()
    def forward(self, x, y):
#        print(x.shape)###-1,1,80,192
        return self.dis1(x, y)

    def Reg_loss(self,x):
        out=self.dis1.Reg_out(x)

        mean_center = torch.mean(out, 0, keepdim=True)
        centerloss = torch.sum(torch.square(out - mean_center)) / 20
        # centerloss=torch.mean(torch.var(out,0))
        Z = torch.unsqueeze(out - mean_center, 2)

        Cov = torch.mean(torch.matmul(Z, torch.permute(Z, [0, 2, 1])), 0)
        DeCov = (torch.sum(torch.square(Cov)) - torch.sum(torch.square(torch.diag(Cov, 0)))) * 0.5
        var_loss = torch.mean(F.relu(1 - torch.std(out, 1)))
        Reg_loss = DeCov + var_loss + centerloss
        return Reg_loss
    def classifier(self, x,y):
        x=self.cls1.get_feature(x)
        #print(x.shape)
        loss, _ = self.AAMsoftmax(x, y)
        return x,loss

class AAMsoftmax(nn.Module):
    def __init__(self, n_class=60, m=0.2, s=30):
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 60), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        #loss = self.ce(output, label)
        #prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return output, output

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=60, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        self.main = nn.Sequential(*blocks)
        self.out = nn.Conv2d(dim_out, num_domains, 1, 1, 0)
    def get_feature(self, x):
        out = self.out(self.main(x))
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out

    def Reg_out(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)
        return out
    def forward(self, x, y):
        out2=self.get_feature(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out2 = out2[idx, y]  # (batch)
        return out2


def build_model(args, F0_model, ASR_model):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
        
    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model)
    
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema