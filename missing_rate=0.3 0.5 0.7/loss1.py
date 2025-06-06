import torch.nn as nn
import math
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device, lambda1 = 0.5, lambda2 = 0.5):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim = 2)
        self.criterion = nn.CrossEntropyLoss(reduction = "none")  # 支持加权

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j, confidence=None):
        N = 2 * h_i.size(0)
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, h_i.size(0))
        sim_j_i = torch.diag(sim, -h_i.size(0))
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        loss = self.criterion(logits, labels)
        if confidence is not None:
            loss = loss * confidence.repeat(2)
        return loss.mean()

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j
        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        return loss.mean() + entropy

    def forward(self, hs, qs, confidence=None):
        loss_feat = 0
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                loss_feat += self.forward_feature(hs[i], hs[j], confidence=confidence)
        loss_label = 0
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                loss_label += self.forward_label(qs[i], qs[j])
        return self.lambda1 * loss_feat + self.lambda2 * loss_label
