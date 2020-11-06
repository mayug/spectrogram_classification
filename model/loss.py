import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    # print('sizes ', [output.shape, target.shape])
    # print('types ', [type(output.data[0]), type(target.data[0])])
    # print('values ', [output.data[0], target.data[0]])

    return F.mse_loss(output, target)