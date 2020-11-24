import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import torchvision.models as models
from .squeeze_net import squeezenet1_1

class SimpleClassificationModel(BaseModel):
    def __init__(self, num_classes=1, num_channels=1, regression=False):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.adaptive_pool2d = nn.AdaptiveAvgPool2d((5,5))
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, num_classes)

        # initialize the biases of the final layer 
        # for the rx_pose_carry task only
        self.fc2.bias.data = torch.Tensor( [-1.0, -0.61, -2.48, -3.28])
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x= self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.adaptive_pool2d(x)
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # print('after fc2 ', x.shape)
        # print('after doftmax', F.log_softmax(x, dim=1).shape)
        return F.log_softmax(x, dim=1)


class SimpleRegressionModel(BaseModel):
    def __init__(self, num_classes=1,num_channels=1, regression=True):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(480, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.adaptive_avg_pool2d(self.conv3_drop(self.conv3(x)), (4, 4)))
        x = x.view(-1, 480)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x.squeeze()

class resnet18(BaseModel):
    def __init__(self, 
                num_classes=1,
                num_channels=1,
                regression=False):
        super().__init__()
        self.regression = regression
        self.model = models.resnet18(pretrained=False, 
                                    num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(num_channels, 64, 
                                kernel_size=7, stride=2,
                                padding=3,
                                bias=False)
        # initialize the biases of the final layer for the rx_pose_carry task
        if not self.regression:
            self.model.fc.bias.data = torch.Tensor( [-1.0, -0.61, -2.48, -3.28])
    def forward(self, x):
        x = self.model(x)
        if self.regression:
            return x.squeeze()
        else:
            return F.log_softmax(x, dim=1)

class squeezenet(BaseModel):
    def __init__(self, 
                num_classes=1,
                num_channels=1,
                regression=False):
        super().__init__()
        self.regression = regression
        self.model = squeezenet1_1(pretrained=False, 
                                   num_classes=num_classes,
                                   num_channels=num_channels)

  
    def forward(self, x):
        x = self.model(x)
        if self.regression:
            return x.squeeze()
        else:
            return F.log_softmax(x, dim=1)


def distance(Ptx, Rssi, P0=61, n=2.8):

    distance = 10**((Ptx - Rssi-P0)/(10*n))
    return distance


class Simple1DCNN(torch.nn.Module):
    # takes in ble_rssi and predicts P0 and n
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=1, out_channels=20, kernel_size=5, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=2, kernel_size=1)
        self.layer2.bias.data = torch.Tensor([61, 2.8])
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        return x.squeeze()


class DistanceModel(torch.nn.Module):
    def __init__(self, regression):
        super(DistanceModel, self).__init__()
        self.inter_model = Simple1DCNN()
    def forward(self, x):
        # print(x.shape)
        # print('type ', [type(x.cpu().detach().numpy()[0,0,0])])
        # print(x[:, 0, :].shape)
        
        inter = self.inter_model(x[:, 0, :].unsqueeze(1))
        # print('inter ', inter.shape)
        P0 = inter[:, 0]
        n = inter[:, 1]
        # print('ptx', x[:, 1, :].mean(dim=1).shape)
        # print('rsssi', x[:, 0, :].mean(dim=1).shape)
        # print('P0', P0.shape)
        # print('n', n.shape)
        d = distance(x[:, 1, :].mean(dim=1), x[:, 0, :].mean(dim=1), P0, n)
        # print('type pf d ', type(d.cpu().detach().numpy()[0]))
        return d

