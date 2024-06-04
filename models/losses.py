import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1., p=2, eps=1e-6, reduction='mean'):
        super(TripletLoss, self).__init__()

        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p, eps=eps, reduction=reduction)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)


class TripletClassifierLoss(nn.Module):
    def __init__(self):
        super(TripletClassifierLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, d_ap, d_an, d_pn):
        y_near = torch.ones_like(d_ap)
        y_far = torch.zeros_like(d_an)

        # bce_aa = self.bce(d_aa, y_near)
        bce_ap = self.bce(d_ap, y_near)
        bce_an = self.bce(d_an, y_far)
        bce_pn = self.bce(d_pn, y_far)

        return torch.mean(bce_ap + bce_an + bce_pn)
