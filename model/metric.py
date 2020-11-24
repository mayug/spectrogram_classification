import torch
from sklearn import metrics
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def mse(output, target):
    with torch.no_grad():
        output_np = output.cpu().numpy()
        target_np = target.cpu().numpy()
        # print('output_np ', [output_np.shape, output_np.min(), output_np.max()])
        # print('target_np ', [target_np.shape, target_np.min(), target_np.max()])
        # get rid of this
        # output_np[np.isnan(output_np)] = 0

        mse = metrics.mean_squared_error(target_np, output_np)
    return mse


def r2_score(output, target):
    with torch.no_grad():
        output_np = output.cpu().numpy()
        target_np = target.cpu().numpy()
        # get rid of this
        # output_np[np.isnan(output_np)] = 0

        mse = metrics.r2_score(target_np, output_np)
    return mse