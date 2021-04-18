import math
import torch
import torch.nn as nn
from thop import profile
from rdwc import depthwise_separable_conv
from config import device
import torchvision


class VGG5(nn.Module):
    def __init__(self, n_classes=47, input_ch=3):
        super().__init__()

        self.input_ch = input_ch
        self.n_classes = n_classes
        self.dropout_p = 0.5

        self.conv = nn.Sequential(nn.Conv2d(input_ch, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2, stride=2),

                                  nn.Conv2d(64, 128, 3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2, stride=2),

                                  nn.Conv2d(128, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2, stride=2),

                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2, stride=2),
                                  )

        self.fc = nn.Sequential(nn.Dropout(self.dropout_p),
                                nn.Linear(256, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.dropout_p),
                                nn.Linear(512, self.n_classes),
                                )

        print("num_params : ", self.count_parameters())
        self.init_conv2d()

    def init_conv2d(self):
        for m in self.conv.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size()[0], -1)  # [B, 47, 1]
        x = self.fc(x)
        return x


class VGG5_DWC(nn.Module):
    def __init__(self, n_classes=47, input_ch=3, pruning_ratio=1., relative=False):
        super().__init__()

        self.relative = relative
        self.input_ch = input_ch
        self.n_classes = n_classes
        self.pruning_ratio = pruning_ratio
        self.dropout_p = 0.5

        self.pruned_dwc_ = nn.Sequential(depthwise_separable_conv(self.input_ch, 1,
                                                                  int(64 * self.pruning_ratio)),
                                         nn.BatchNorm2d(int(64 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         depthwise_separable_conv(int(64 * self.pruning_ratio), 1,
                                                                  int(64 * self.pruning_ratio), self.relative),
                                         nn.BatchNorm2d(int(64 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(2, stride=2),

                                         # conv2
                                         depthwise_separable_conv(int(64 * self.pruning_ratio), 1,
                                                                  int(128 * self.pruning_ratio), self.relative),
                                         nn.BatchNorm2d(int(128 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         depthwise_separable_conv(int(128 * self.pruning_ratio), 1,
                                                                  int(128 * self.pruning_ratio), self.relative),
                                         nn.BatchNorm2d(int(128 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(2, stride=2),

                                         # conv3
                                         depthwise_separable_conv(int(128 * self.pruning_ratio), 1,
                                                                  int(256 * self.pruning_ratio), self.relative),
                                         nn.BatchNorm2d(int(256 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         depthwise_separable_conv(int(256 * self.pruning_ratio), 1,
                                                                  int(256 * self.pruning_ratio), self.relative),
                                         nn.BatchNorm2d(int(256 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         depthwise_separable_conv(int(256 * self.pruning_ratio), 1,
                                                                  int(256 * self.pruning_ratio), self.relative),
                                         nn.BatchNorm2d(int(256 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(2, stride=2),

                                         # conv4
                                         depthwise_separable_conv(int(256 * self.pruning_ratio), 1,
                                                                  int(256 * self.pruning_ratio), self.relative),
                                         nn.BatchNorm2d(int(256 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         depthwise_separable_conv(int(256 * self.pruning_ratio), 1,
                                                                  int(256 * self.pruning_ratio), self.relative),
                                         nn.BatchNorm2d(int(256 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         depthwise_separable_conv(int(256 * self.pruning_ratio), 1,
                                                                  int(256 * self.pruning_ratio), self.relative),
                                         nn.BatchNorm2d(int(256 * self.pruning_ratio)),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(2, stride=2),
                                         )

        self.pruned_fc = nn.Sequential(nn.Dropout(self.dropout_p),
                                       nn.Linear(int(256 * self.pruning_ratio), 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(self.dropout_p),
                                       nn.Linear(512, self.n_classes),
                                       )

        print("num_params : ", self.count_parameters())
        self.init_conv2d()

    def init_conv2d(self):
        for m in self.pruned_dwc_.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        x = self.pruned_dwc_(x)
        x = x.view(x.size()[0], -1)  # [B, 47, 1]
        x = self.pruned_fc(x)
        return x


class VGG16(nn.Module):
    def __init__(self, pretrained=False, n_classes=10, input_ch=3):
        super().__init__()
        self.input_ch = input_ch
        self.n_classes = n_classes
        self.dropout_p = 0.5
        self.features = nn.Sequential(nn.Conv2d(self.input_ch, 64, 3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      # conv1

                                      nn.Conv2d(64, 128, 3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      # conv2

                                      nn.Conv2d(128, 256, 3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      # conv3

                                      nn.Conv2d(256, 512, 3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      # conv4

                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      # conv5
                                      )
        self.fc = nn.Sequential(nn.Dropout(self.dropout_p),
                                nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.dropout_p),
                                nn.Linear(512, self.n_classes),
                                )

        print("num_params : ", self.count_parameters())
        self._initialize_weights()

        if pretrained:

            std = torchvision.models.vgg16_bn(pretrained=True).features.state_dict()
            model_dict = self.features.state_dict()
            pretrained_dict = {k: v for k, v in std.items() if k in model_dict}  # 여기서 orderdict 가 아니기 때문에
            model_dict.update(pretrained_dict)
            self.features.load_state_dict(model_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # [B, 47, 1]
        x = self.fc(x)
        return x


class VGG16_DWC(nn.Module):
    def __init__(self, n_classes=10, input_ch=3, pruning_ratio=1., relative=False):
        super().__init__()

        self.relative = relative
        self.input_ch = input_ch
        self.n_classes = n_classes
        self.pruning_ratio = pruning_ratio
        self.dropout_p = 0.5

        self.features = nn.Sequential(depthwise_separable_conv(self.input_ch, 1,
                                                               int(64 * self.pruning_ratio)),
                                      nn.BatchNorm2d(int(64 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      depthwise_separable_conv(int(64 * self.pruning_ratio), 1,
                                                               int(64 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(64 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv1

                                      depthwise_separable_conv(int(64 * self.pruning_ratio), 1,
                                                               int(128 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(128 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      depthwise_separable_conv(int(128 * self.pruning_ratio), 1,
                                                               int(128 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(128 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv2

                                      depthwise_separable_conv(int(128 * self.pruning_ratio), 1,
                                                               int(256 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(256 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      depthwise_separable_conv(int(256 * self.pruning_ratio), 1,
                                                               int(256 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(256 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      depthwise_separable_conv(int(256 * self.pruning_ratio), 1,
                                                               int(256 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(256 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv3

                                      depthwise_separable_conv(int(256 * self.pruning_ratio), 1,
                                                               int(512 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(512 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      depthwise_separable_conv(int(512 * self.pruning_ratio), 1,
                                                               int(512 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(512 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      depthwise_separable_conv(int(512 * self.pruning_ratio), 1,
                                                               int(512 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(512 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv4

                                      depthwise_separable_conv(int(512 * self.pruning_ratio), 1,
                                                               int(512 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(512 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      depthwise_separable_conv(int(512 * self.pruning_ratio), 1,
                                                               int(512 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(512 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      depthwise_separable_conv(int(512 * self.pruning_ratio), 1,
                                                               int(512 * self.pruning_ratio), self.relative),
                                      nn.BatchNorm2d(int(512 * self.pruning_ratio)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      nn.Dropout2d(self.dropout_p),
                                      # conv5

                                      )
        self.fc = nn.Sequential(nn.Dropout(self.dropout_p),
                                nn.Linear(int(512 * self.pruning_ratio), 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.dropout_p),
                                nn.Linear(512, self.n_classes),
                                )

        print("num_params : ", self.count_parameters())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # [B, 47, 1]
        x = self.fc(x)
        return x


def build_model(opts):

    assert opts.dataset_type in ['mnist', 'fashionmnist', 'cifar10']

    model = None
    pruned_model = None

    if opts.dataset_type == 'mnist' or opts.dataset_type == 'fashionmnist':
        model = VGG5_DWC(opts.n_classes, opts.input_ch, pruning_ratio=1.0, relative=True).to(device)
        pruned_model = VGG5_DWC(opts.n_classes, opts.input_ch, opts.pruning_ratio, True).to(device)

    elif opts.dataset_type == 'cifar10':
        model = VGG5_DWC(opts.n_classes, opts.input_ch, pruning_ratio=1.0, relative=True).to(device)
        pruned_model = VGG16_DWC(opts.n_classes, opts.input_ch, opts.pruning_ratio, True).to(device)

    return model, pruned_model


if __name__ == '__main__':

    batch_size = 2

    # ------------------------ for mnist ----------------------------
    input = torch.randn([batch_size, 1, 28, 28])
    model = VGG5_DWC(n_classes=10, input_ch=1, pruning_ratio=0.875, relative=True)

    macs, params = profile(model, inputs=(input,))
    print("flops :", macs / batch_size)
    print("params : ", params)

    '''
    0.875    
    flops : 16428592.0
    params :  452044.0
    
    0.75
    flops : 12412480.0
    params :  351940.0    
    
    0.5
    flops : 6051424.0
    params :  188212.0
    '''

    # ------------------------ for cifar 10 ----------------------------
    input = torch.randn([batch_size, 3, 32, 32])
    model = VGG16_DWC(n_classes=10, input_ch=3, pruning_ratio=0.5, relative=True)

    macs, params = profile(model, inputs=(input,))
    print("flops :", macs / batch_size)
    print("params : ", params)

    '''
    0.875    
    flops : 30687232.0
    params :  1560304.0

    0.75
    flops : 23064576.0
    params :  1185144.0

    0.5
    flops : 11063296.0
    params :  588040.0
    '''