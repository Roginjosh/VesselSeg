import torch
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()
        self.c1 = conv_block(in_ch, base);   self.p1 = nn.MaxPool2d(2)
        self.c2 = conv_block(base, base*2);  self.p2 = nn.MaxPool2d(2)
        self.c3 = conv_block(base*2, base*4);self.p3 = nn.MaxPool2d(2)
        self.c4 = conv_block(base*4, base*8);self.p4 = nn.MaxPool2d(2)
        self.c5 = conv_block(base*8, base*16)

        self.u6 = nn.ConvTranspose2d(base*16, base*8, 2, 2); self.c6 = conv_block(base*16, base*8)
        self.u7 = nn.ConvTranspose2d(base*8,  base*4, 2, 2); self.c7 = conv_block(base*8,  base*4)
        self.u8 = nn.ConvTranspose2d(base*4,  base*2, 2, 2); self.c8 = conv_block(base*4,  base*2)
        self.u9 = nn.ConvTranspose2d(base*2,  base,   2, 2); self.c9 = conv_block(base*2,  base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))
        c4 = self.c4(self.p3(c3))
        c5 = self.c5(self.p4(c4))

        x  = self.u6(c5); x = torch.cat([x, c4], 1); x = self.c6(x)
        x  = self.u7(x);  x = torch.cat([x, c3], 1); x = self.c7(x)
        x  = self.u8(x);  x = torch.cat([x, c2], 1); x = self.c8(x)
        x  = self.u9(x);  x = torch.cat([x, c1], 1); x = self.c9(x)
        return self.out(x)  # logits
