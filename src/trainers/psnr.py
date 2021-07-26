import torch

def _psnr(img1, img2, mse_loss_func, channel):
    rmse = torch.sqrt(mse_loss_func(img1, img2))
    # max_value = torch.max(torch.cat((img1, img2)))
    max_value = torch.max(img1)
    # max_value = 1.0
    # print('max_value',max_value.cpu().numpy())
    # print('rmse',rmse.cpu().numpy())
    psnr = 20*torch.log10(max_value/rmse)
    # print('psnr',psnr.cpu().numpy())
    return psnr

class PSNR(torch.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.channel = 1
        self.mse_loss_func = torch.nn.MSELoss(reduction='mean')

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        self.channel = channel
        if img1.is_cuda:
            self.mse_loss_func = self.mse_loss_func.cuda(img1.get_device())
        return _psnr(img1, img2, self.mse_loss_func, self.channel)

def psnr(img1, img2, size_average=True):
    (_, channel, _, _) = img1.size()
    mse_loss_func = torch.nn.MSELoss( reduction='mean')
    if img1.is_cuda:
        mse_loss_func = mse_loss_func.cuda(img1.get_device())
    return _psnr(img1, img2, mse_loss_func, channel)
