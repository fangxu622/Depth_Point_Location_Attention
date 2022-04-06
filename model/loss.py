
import torch 
from torch import nn
import torch.nn.functional as F

__all__ = ['pose_loss', 'standard_pose_loss']

class pose_loss(nn.Module):
    def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
        super(pose_loss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -6.25

        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

        # if learn_beta:
        #     self.sx.requires_grad = True
        #     self.sq.requires_grad = True
        #
        # self.sx = self.sx.to(device)
        # self.sq = self.sq.to(device)

        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        pred_q = F.normalize(pred_q, p=2, dim=1)
        loss_x = F.l1_loss(pred_x, target_x)
        loss_q = F.l1_loss(pred_q, target_q)

        loss = torch.exp(-self.sx) * loss_x \
               + self.sx \
               + torch.exp(-self.sq) * loss_q \
               + self.sq

        #self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()

class standard_pose_loss(nn.Module):
    def __init__(self, cfg):
        super(standard_pose_loss, self).__init__()

        self.beta = nn.Parameter(torch.Tensor([cfg.beta]), requires_grad=False)

    def forward(self, pred_x, pred_q, target_x, target_q):
        
        pred_q = F.normalize(pred_q, p=2, dim=1)
        loss_x = F.mse_loss(pred_x, target_x)
        loss_q = self.beta * F.mse_loss(pred_q, target_q)

        loss = loss_x + loss_q

        #self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()


