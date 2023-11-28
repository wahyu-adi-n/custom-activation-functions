import torch.nn as nn
# larger resnet inspired model to test if my functions scale well to deep models like ReLU does
# create the ResidualBlock for use in the ResNet

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        #my chosen activation function
                        nn.ReLU()
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels)
                    )
        self.downsample = downsample
        self.act_func = nn.ReLU()
        self.out_channels = out_channels
            
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.act_func(out)
        return out
            
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
            
        self.inplanes  = 64
        self.conv1 = nn.Sequential(
                        # FashionMNIST dataset images are greyscale so 1 input channel
                        nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                    )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
                    
        #had to change to 3 cos the FashionMNIST images start off small and would be too small(less than 0 in size) if let at 7
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(512, num_classes)
            
            
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                            nn.BatchNorm2d(planes)
                        )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
                    
        #need to uncomment these lines and previous lines if I want to have the intended 4 layers
        # x = self.layer2(x)
        # x = self.layer3(x)
            
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
            
        return x