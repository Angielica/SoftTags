import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1., p=2, eps=1e-6, reduction='mean'):
        super(TripletLoss, self).__init__()

        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p, eps=eps, reduction=reduction)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)

