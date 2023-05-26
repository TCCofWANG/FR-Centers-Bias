import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CentersBiasFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, m_m=0.05, momentum=0.99, constant_t=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self.m_m = m_m

        self.constant_t = constant_t

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        center = torch.FloatTensor(out_features, in_features)
        nn.init.xavier_uniform_(center)
        self.register_buffer('center', center)

        self.register_buffer('t_ada', torch.zeros(1))

        self.momentum = momentum

    def forward(self, embeddings, label, **kwargs):
        cos_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        with torch.no_grad():
            one_hot = torch.zeros_like(cos_theta, requires_grad=False)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            # update centers
            self.center[label] = embeddings * (1 - self.momentum) + self.momentum * self.center[label]

            # update t
            target_cos = one_hot * cos_theta
            not_zero = (target_cos != 0).float()
            sum_cos = torch.sum(target_cos)
            not_zero_cnt = torch.sum(not_zero)

            update_value = sum_cos / not_zero_cnt

            self.t_ada = update_value * (1 - self.momentum) + self.momentum * self.t_ada
            if self.constant_t:
                t_ada = 1.
            else:
                t_ada = self.t_ada

            # calculate hard rate
            hard_rate = 1 - F.cosine_similarity(self.weight, self.center)
            norm_hard_rate = (hard_rate - hard_rate.min()) / (hard_rate.max() - hard_rate.min())

            # calculate margin
            margin = self.m + norm_hard_rate * t_ada * self.m_m
            margin = margin[None, :]
            margin = torch.clamp(margin, self.m, self.m + self.m_m)

        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(0, 1))

        cos_m = torch.cos(margin)
        sin_m = torch.sin(margin)
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        th = torch.cos(math.pi - margin)
        mm = torch.sin(math.pi - margin) * margin

        cos_theta_m = torch.where(cos_theta > th, cos_theta_m, cos_theta - mm)

        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output *= self.s

        loss = F.cross_entropy(output, label)

        return {'total_loss': loss, 't': t_ada,
                'm_mean': margin.mean(), 'm_max': margin.max(), 'm_min': margin.min(), }
