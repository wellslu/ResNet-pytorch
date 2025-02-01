import mlconfig
from torch import nn
import torch.nn.functional as F

archs = {18: [[64, 64, 2], [64,128,2], [128,256,2], [256,512,2]],
         34: [[64, 64, 3], [64,128,4], [128,256,6], [256,512,3]],
        #  50: [[64, 64, 3], [256,128,4], [512,256,6], [1024,512,3]],
        #  101: [[64, 64, 3], [256,128,4], [512,256,23], [1024,512,3]],
        #  152: [[64, 64, 3], [256,128,8], [512,256,36], [1024,512,3]]
         }

class ConvBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        super(ConvBN, self).__init__(*layers)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.CB1 = ConvBN(in_channels, out_channels, stride=stride)
        self.CB2 = ConvBN(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = None
        
    def forward(self, x):
        out = self.CB1(x)
        out = F.relu(out)
        out = self.CB2(out)

        if self.shortcut is not None:
            x = self.shortcut(x)
        out += x
        out = F.relu(out)

        return out
        
# first layer do shortcut, last layer do out_channels*4
# class BottleneckBlock(nn.Module):
    
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BottleneckBlock, self).__init__()
#         self.CB1 = ConvBN(in_channels, out_channels, kernel_size=1, padding=0)
#         self.CB2 = ConvBN(out_channels, out_channels, stride=stride)
#         self.CB3 = ConvBN(out_channels, out_channels, kernel_size=1, padding=0)
#         if in_channels != out_channels:
#             self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1, stride=stride)
#         else:
#             self.shortcut = None
        
#     def forward(self, x):
#         out = self.CB1(x)
#         out = F.relu(out)
#         out = self.CB2(out)
#         out = F.relu(out)
#         out = self.CB3(out)

#         if self.shortcut is not None:
#             x = self.shortcut(x)

#         out += x
#         out = F.relu(out)

#         return out
    
        
        
@mlconfig.register
class ResNet(nn.Module):

    def __init__(self, arch, num_classes=10):
        super(ResNet, self).__init__()
        arch_list = archs[arch]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # self.block = ResidualBlock if arch in [18, 34] else BottleneckBlock
        self.block = ResidualBlock
        
        self.stage1 = self._make_layer(arch_list[0][0], arch_list[0][1], arch_list[0][2], stage1=True)
        self.stage2 = self._make_layer(arch_list[1][0], arch_list[1][1], arch_list[1][2])
        self.stage3 = self._make_layer(arch_list[2][0], arch_list[2][1], arch_list[2][2])
        self.stage4 = self._make_layer(arch_list[3][0], arch_list[3][1], arch_list[3][2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(arch_list[3][1] if arch in [18, 34] else arch_list[3][1]*4 , num_classes),
            nn.Softmax(dim=1)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _make_layer(self, in_channels, out_channels, block_num, stage1=False):
        layers = []
        for b in range(block_num):
            if b == 0 and not stage1:
                layers.append(self.block(in_channels, out_channels, stride=2))
            else:
                layers.append(self.block(in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(*layers)
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
