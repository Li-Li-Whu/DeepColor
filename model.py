import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Conv2DTranspose, BatchNorm
from paddle.nn import ReLU, MaxPool2D, Sequential, Dropout, Linear


class ConvBlock(fluid.dygraph.Layer):
    def __init__(self, inchannels, outchannels, batch_norm=False, pool=True):
        super(ConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.pool = pool

        self.conv1 = Conv2D(inchannels, outchannels, filter_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.norm1 = BatchNorm(outchannels)
        self.conv2 = Conv2D(outchannels, outchannels, filter_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        if batch_norm:
            self.norm1 = BatchNorm(outchannels)
        if pool:
            self.pool1 = MaxPool2D(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        if self.batch_norm:
            x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        if self.batch_norm:
            x = self.norm2(x)
        if self.pool:
            x = self.pool1(x)

        return x

class AlexNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=1000, batch_norm=False):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes

        self.feature = Sequential(
            ConvBlock(6, 64, batch_norm),
            ConvBlock(64, 64, batch_norm),
            ConvBlock(64, 128, batch_norm),
            ConvBlock(128, 128, batch_norm, pool=False),
        )
        self.fc = Sequential(
            Dropout(0.5),
            Linear(128 * 16 * 16, 1024),
            ReLU(),
            Dropout(0.5),
            Linear(1024, num_classes)
        )

    def forward(self, x):
        # print(111)
        # x = self.features(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.classifier(x)
        x = self.feature(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.fc(x)
        return x