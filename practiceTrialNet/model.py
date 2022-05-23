import torch
from torch import nn
from torch.nn import functional as F
from config import HP

def mish(x):  #[N, ...]
    return x*torch.tanh(F.softplus(x))

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return mish(x)

class DSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DSConv2d, self).__init__()
        assert kernel_size%2 ==1, "odd needed"
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size//2, kernel_size//2),
            groups= in_channels
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,1)
        )

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.pointwise_conv(out)
        return out

class TrialNet(nn.Module):
    def __init__(self, in_channel):
        super(TrialNet, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_channel),
            Mish(),
            DSConv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3),
            nn.BatchNorm2d(in_channel),
            Mish(),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(7, 7), padding=(7//2, 7//2))
        )
        self.right = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(7, 7), padding=(7//2, 7//2)),
            nn.BatchNorm2d(in_channel),
            Mish(),
            DSConv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3),
            nn.BatchNorm2d(in_channel),
            Mish(),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(1, 1)),
        )

    def forward(self, x):
        out_left = self.left(x)
        out_right = self.right(x)
        out = out_left+out_right+x
        return mish(out)

class FinalTrialNet(nn.Module):
    def __init__(self):
        super(FinalTrialNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=HP.data_channels, out_channels=64, kernel_size=(3, 3), padding=(3//2, 3//2)),
            nn.BatchNorm2d(64),
            Mish(),
            TrialNet(in_channel=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(3//2, 3//2)),
            nn.BatchNorm2d(128),
            Mish(),
            TrialNet(in_channel=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            nn.BatchNorm2d(256),
            Mish(),
            TrialNet(in_channel=256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            TrialNet(in_channel=256),
            TrialNet(in_channel=256),
            TrialNet(in_channel=256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        ) # original input shape: [N, 3, 112, 112] -> [N, 256, 7, 7]

        self.fc = nn.Sequential(
            nn.Linear(in_features=256*7*7, out_features=2048),
            Mish(),
            nn.Dropout(HP.fc_drop_prob),
            nn.Linear(in_features=2048, out_features=1024),
            Mish(),
            nn.Dropout(HP.fc_drop_prob),
            nn.Linear(in_features=1024, out_features=HP.classes_num)
        )

    def forward(self, x):
        out = self.conv(x) # [N, 256, 7, 7]
        out = out.view( x.size(0), -1)  # fc input size [N, *, dim],
        out = self.fc(out)  #-> [N, 256*7*7]
        return out

if __name__ == '__main__':
    model = FinalTrialNet()
    x = torch.randn(size=(5, 3, 112, 112),)
    ypred = model(x)
    print(ypred.size())  # [N, classes_num]
