import torch
import torch.nn as nn


class depthwise_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class pointwise_conv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.pointwise(x)
        return out


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, relative=False):
        super(depthwise_separable_conv, self).__init__()
        self.relative = relative

        if self.relative == False:
            self.dwc = nn.Sequential(nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin),
                                     nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1),
                                     )

        else:
            self.dwc = nn.Sequential(nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=int(nin/2)),
                                     nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1),
                                     )

    def forward(self, x):
        out = self.dwc(x)
        return out


if __name__ == '__main__':

    img = torch.randn([2, 1, 28, 28])
    model = nn.Sequential(depthwise_separable_conv(1, 1, 3),
                          nn.ReLU())
    print(model(img).size())