import torch
import torch.nn as nn

from torchvision.models import resnet50


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128, n_channels=3, h_size=64, w_size=64, last_dim=64):
        super(EmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim

        self.last_dim = last_dim
        self.n_channels = n_channels
        self.h_size = h_size
        self.w_size = w_size

        # self.resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        out = self.features(torch.rand(1, self.n_channels, self.h_size, self.w_size))
        self.size1 = out.shape[2]
        self.size2 = out.shape[3]

        self.fc = nn.Sequential(
            nn.Linear(self.last_dim * self.size1 * self.size2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.embedding_dim)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128, n_channels=3, h_size=64, w_size=64, eps=.99):
        super(SiameseNetwork, self).__init__()
        self.eps = eps
        self.embedder = EmbeddingNet(embedding_dim=embedding_dim, n_channels=n_channels,
                                     h_size=h_size, w_size=w_size)

        self.fc = nn.Sequential(
            nn.Linear(in_features=2 * embedding_dim, out_features=embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2),
            nn.LeakyReLU(),
            nn.Linear(in_features=embedding_dim//2, out_features=embedding_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(in_features=embedding_dim//4, out_features=1),
            nn.Sigmoid()
        )

    def euclidean_distance(self, emb1, emb2):
        dis = ((emb1 - emb2) ** 2.).sum(dim=1, keepdim=True).sqrt()
        return dis

    def inference(self, x1, x2):
        emb1 = self.embedder(x1)
        emb2 = self.embedder(x2)

        pd = self.fc(torch.cat((emb1, emb2), dim=1))
        # d = self.euclidean_distance(emb1, emb2)
        return pd

    def forward(self, a, p, n):
        a_emb = self.embedder(a)
        p_emb = self.embedder(p)
        n_emb = self.embedder(n)

        # d_aa = self.euclidean_distance(a_emb, a_emb)
        p_d_ap = self.fc(torch.cat((a_emb, p_emb), dim=1))
        p_d_an = self.fc(torch.cat((a_emb, n_emb), dim=1))
        # d_pn = self.euclidean_distance(p_emb, n_emb)

        return p_d_ap, p_d_an


