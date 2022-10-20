import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models import l2_norm

class ArcFace(nn.Module):
    def __init__(self, embed_size=512, num_classes=10572, s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.num_classes = num_classes
        self.m = m
        self.s = s
        
        self.kernel = nn.Parameter(torch.Tensor(embed_size, num_classes))
        self.kernel.data.uniform_(-1,1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.mm = self.sin_m * m
        self.th = math.cos(math.pi - m)

    def forward(self, embeddings, label):
        # l2 norm weights
        n_embeddings = len(embeddings)
        kernel_norm = l2_norm(self.kernel, axis=0)

        #cos(theta+m)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1,1)
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2))
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.th
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 
        idx_ = torch.arange(0, n_embeddings, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]

        output = output * self.s
        return output

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

if __name__ == "__main__":
    
    sample = torch.rand(3, 512)

    label = torch.ones(3,1).long()

    arcface = ArcFace()

    output = arcface(sample, label)

    print(output.shape)
