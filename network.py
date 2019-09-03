from network_parts import *


class UNetFB(nn.Module):
    # based on https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    def __init__(self):
        super(UNetFB, self).__init__()
        self.inc = Inconv(1, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)
        self.outc = Outconv(16, 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
