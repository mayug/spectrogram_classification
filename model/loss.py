import torch.nn.functional as F
from data_loader.mixup import mixup_criterion

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    # print('sizes ', [output.shape, target.shape])
    # print('types ', [type(output.data[0]), type(target.data[0])])
    # print('values ', [output.data[0], target.data[0]])
    # print('type ', [type(output.cpu().detach().numpy()[0])])
    # print('type ', [type(target.cpu().detach().numpy()[0])])
    return F.mse_loss(output, target)


def mixup_mse_loss(output, target_a, target_b, lam):
    loss = mixup_criterion(mse_loss, output, target_a, target_b, lam)
    return loss 