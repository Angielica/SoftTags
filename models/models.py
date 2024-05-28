import torch
import torch.nn as nn

from torchvision.models import resnet50

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim

        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.embedding_dim)
        )

    def forward(self, x):
        return self.resnet(x)


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.embedder = EmbeddingNet(embedding_dim=embedding_dim)

    def euclidean_distance(self, emb1, emb2):
        dis = ((emb1 - emb2) ** 2.).sum(dim=1, keepdim=True).sqrt()
        return dis

    def forward(self, a, p, n):
        a_emb = self.embedder(a)
        p_emb = self.embedder(p)
        n_emb = self.embedder(n)

        d_aa = self.euclidean_distance(a_emb, a_emb)
        d_ap = self.euclidean_distance(a_emb, p_emb)

        d_an = self.euclidean_distance(a_emb, n_emb)
        d_pn = self.euclidean_distance(p_emb, n_emb)

        return torch.exp(-d_aa), torch.exp(-d_ap), torch.exp(-d_an), torch.exp(-d_pn)


