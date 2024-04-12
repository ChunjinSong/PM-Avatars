import torch
import torch.nn as nn

def img2mse(img, target):
    return (img - target).pow(2.).mean()

def mse2psnr(mse):
    return  -10. * torch.log10(mse) / torch.log10(torch.Tensor([10.]))

def img2psnr(img, target):
    return mse2psnr(img2mse(img, target))

class BaseLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class NeRFRGBLoss(BaseLoss):

    def __init__(self, fine=1.0, coarse=1.0, **kwargs):
        super(NeRFRGBLoss, self).__init__(**kwargs)
        self.fine = fine
        self.coarse = coarse

    def forward(self, batch, preds, base_bg=1.0, **kwargs):

        rgb_pred = preds['rgb_map']
        loss_fine = (rgb_pred - batch['target_s']).abs().mean()

        loss_coarse = torch.tensor(0.0)

        if 'rgb0' in preds:
            rgb_pred = preds['rgb0']
            loss_coarse = (rgb_pred - batch['target_s']).abs().mean()
        
        loss = loss_fine * self.fine + loss_coarse * self.coarse

        return loss, {'rgbL': loss.item(), 'loss_fine': loss_fine.item(), 'loss_coarse': loss_coarse.item()}

class SigmaLoss(BaseLoss):

    def forward(self, batch, preds, **kwargs):
        vol_scale = preds['vol_scale']
        scale_loss = (torch.prod(vol_scale, dim=-1)).sum()
        loss = scale_loss * self.weight
        return loss, {'scaleL': loss.item()}