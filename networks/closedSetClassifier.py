"""
    Network definition for standard closed set classifier. 

    Dimity Miller, 2020
"""


import torch
import torchvision
import torch.nn as nn

class closedSetClassifier(nn.Module):
    def __init__(self, num_classes = 20, num_channels = 3, im_size = 64, init_weights = False, dropout = 0.3, **kwargs):
        super(closedSetClassifier, self).__init__()

        self.num_classes = num_classes
        self.encoder = BaseEncoder(num_channels, init_weights, dropout)
        
        if im_size == 32:
            self.classify = nn.Linear(128*4*4, num_classes)
        elif im_size == 64:
            self.classify = nn.Linear(128*8*8, num_classes)
        else:
            print('That image size has not been implemented, sorry.')
            exit()

        if init_weights:
            self._initialize_weights()
        
        self.cuda()


    def forward(self, x):
        batch_size = len(x)

        x = self.encoder(x)
        x = x.view(batch_size, -1)

        outLinear = self.classify(x)

        return outLinear

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class BaseEncoder(nn.Module):
    def __init__(self, num_channels, init_weights, dropout = 0.3, **kwargs): 
        super().__init__()
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.LeakyReLU(0.2)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.conv1 = nn.Conv2d(num_channels,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)


        self.encoder1 = nn.Sequential(
                            self.conv1,
                            self.bn1,
                            self.relu,
                            self.conv2,
                            self.bn2,
                            self.relu,
                            self.conv3,
                            self.bn3,
                            self.relu,
                            self.dropout,
                        )

        self.encoder2 = nn.Sequential(
                                self.conv4,
                                self.bn4,
                                self.relu,
                                self.conv5,
                                self.bn5,
                                self.relu,
                                self.conv6,
                                self.bn6,
                                self.relu,
                                self.dropout,
                            )

        self.encoder3 = nn.Sequential(
                                self.conv7,
                                self.bn7,
                                self.relu,
                                self.conv8,
                                self.bn8,
                                self.relu,
                                self.conv9,
                                self.bn9,
                                self.relu,
                                self.dropout,
                                
                            )

        if init_weights:
            self._initialize_weights()
    
        self.cuda()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        return x3

