from torch import nn

class ResidualSep(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.ReLU(),
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, 
                      padding=0, dilation=dilation, 
                      groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        x = (x / 255.0) * 2.0 - 1.0
        return ((self.model(x) + 1.0) / 2.0) * 255.0

class ResidualHourglass(nn.Module):
    def __init__(self, channels, mult=2):
        super().__init__()

        hidden_channels = int(channels * mult)

        self.blocks = nn.Sequential(
            nn.ReLU(),
            # Downsample
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, stride=2,
                      padding=0, dilation=1, 
                      groups=1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            # Bottleneck
            ResidualSep(channels=hidden_channels, dilation=1),
            ResidualSep(channels=hidden_channels, dilation=2),
            ResidualSep(channels=hidden_channels, dilation=3),
            ResidualSep(channels=hidden_channels, dilation=2),
            ResidualSep(channels=hidden_channels, dilation=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, stride=1,
                      padding=0, dilation=1, 
                      groups=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            # Upsample
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, 
                               padding=0, groups=1, bias=True),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.blocks(x)




class SmallNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()

        self.blocks = nn.Sequential( 
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, width, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(width, affine=True),
            ResidualHourglass(channels=width),
            ResidualHourglass(channels=width),
            ResidualHourglass(channels=width),
            ResidualHourglass(channels=width),
            ResidualSep(channels=width, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )


    def forward(self, x):
        return self.blocks(x)