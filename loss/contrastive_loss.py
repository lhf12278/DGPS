import torch
from torch import nn
import torch.nn.functional as F


# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature):
#         super().__init__()
#         self.T = temperature
#         self.loss = nn.CrossEntropyLoss()
#
#     def forward(self, ft, person_prototype, label):
#         q = ft
#         k = person_prototype[label]
#
#         l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
#         # negative logits: NxK
#         l_neg = torch.einsum('nc,ck->nk', [q, person_prototype.clone().detach().T])
#
#         # logits: Nx(1+K)
#         logits = torch.cat([l_pos, l_neg], dim=1)
#
#         # apply temperature
#         logits /= self.T
#
#         # labels: positive key indicators
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
#         loss = self.loss(logits, labels)
#         return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to("cuda"))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to('cuda'))

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss





