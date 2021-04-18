import time
import os
import torch
from config import device


def test(epoch, model, vis, data_loader, criterion, opts, vis_name='pruning_test', load=False):

    print('Validation of epoch [{}]'.format(epoch))
    model.eval()
    if load:
        state_dict = torch.load(os.path.join(opts.save_path, '{}_{}_{}_{}_{}.pth.tar'.format(opts.save_file_name,
                                                                                             opts.model_name,
                                                                                             opts.dataset_type,
                                                                                             opts.pruning_ratio,
                                                                                             opts.epoch)),
                                map_location=device)
        model.load_state_dict(state_dict)

    correct = 0
    val_avg_loss = 0
    total = 0
    with torch.no_grad():

        for idx, (img, target) in enumerate(data_loader):
            model.eval()
            img = img.to(device)  # [N, 1, 28, 28]
            target = target.to(device)  # [N]
            output = model(img)  # [N, 47]
            loss = criterion(output, target)

            output = torch.softmax(output, dim=1)
            # first eval
            pred, idx_ = output.max(-1)
            correct += torch.eq(target, idx_).sum().item()
            total += target.size(0)
            val_avg_loss += loss.item()

    print('Epoch {} test : '.format(epoch))
    accuracy = correct / total
    print("accuracy : {:.4f}%".format(accuracy * 100.))

    val_avg_loss = val_avg_loss / len(data_loader)
    print("avg_loss : {:.4f}".format(val_avg_loss))
    if vis is not None:
        vis.line(X=torch.ones((1, 2)) * epoch,
                 Y=torch.Tensor([accuracy, val_avg_loss]).unsqueeze(0),
                 update='append',
                 win=vis_name,
                 opts=dict(x_label='epoch',
                           y_label='test_',
                           title='test_loss',
                           legend=['accuracy', 'avg_loss']))

    return accuracy


if __name__ == '__main__':

    from model import VGG16_DWC, VGG5_DWC
    import argparse
    import torchvision.transforms as tfs
    from torchvision.datasets import EMNIST, CIFAR100, CIFAR10, MNIST, FashionMNIST
    from torch.utils.data import DataLoader
    import torch.nn as nn

    # 1. argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=47)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='finetuned_pruned')
    parser.add_argument('--pruned_save_file_name', type=str, default='pruned')
    parser.add_argument('--n_classes', type=int, default=10, help='100, 10, 47, ...')
    parser.add_argument('--batch_size', type=int, default=128)

    #parser.add_argument('--dataset_type', type=str, default='cifar10', help='cifar100, cifar10')        # FIXME
    # parser.add_argument('--dataset_type', type=str, default='mnist', help='cifar100, cifar10')        # FIXME
    parser.add_argument('--dataset_type', type=str, default='fashionmnist', help='cifar100, cifar10')        # FIXME
    #parser.add_argument('--model_name', type=str, default='vgg16')                                      # FIXME
    parser.add_argument('--model_name', type=str, default='vgg5')                                      # FIXME
    parser.add_argument('--pruning_ratio', type=float, default=0.875)                                     # FIXME
    test_opts = parser.parse_args()
    print(test_opts)

    vis = None

    # 4. dataset
    # transform = tfs.Compose([
    #     tfs.Pad(4),
    #     tfs.RandomCrop(32),
    #
    #     tfs.RandomPerspective(),
    #     tfs.RandomRotation(10, fill=(0, 0, 0)),
    #     tfs.RandomHorizontalFlip(),
    #
    #     tfs.ToTensor(),
    #     tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
    #                   std=(0.2023, 0.1994, 0.2010)),
    # ])
    #
    # test_transform = tfs.Compose([tfs.ToTensor(),
    #                               tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
    #                                             std=(0.2023, 0.1994, 0.2010)),
    #                               ])
    #
    # test_set = CIFAR10('./data/CIFAR10',
    #                    train=False,
    #                    download=True,
    #                    transform=test_transform)
    #
    # test_opts.n_classes = 10
    # test_opts.input_ch = 3

    # ---------------------------------- FashionMNIST----------------------------------
    test_transfrom = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.1307,), (0.3081,))
                                  ])

    test_set = FashionMNIST('./data/FashionMNIST',
                            train=False,
                            download=True,
                            transform=test_transfrom)

    test_opts.n_classes = 10
    test_opts.input_ch = 1

    # ---------------------------------- MNIST----------------------------------
    # test_transfrom = tfs.Compose([tfs.ToTensor(),
    #                               tfs.Normalize((0.1307,), (0.3081,))
    #                               ])
    #
    # test_set = MNIST('./data/MNIST',
    #                  train=False,
    #                  download=True,
    #                  transform=test_transfrom)
    #
    # test_opts.n_classes = 10
    # test_opts.input_ch = 1

    # 5. data loader
    test_loader = DataLoader(dataset=test_set,
                             shuffle=True,
                             batch_size=128)

    criterion = nn.CrossEntropyLoss()
    ###################################################
    #                     pruning
    ###################################################
    # 6. model
    # pruned_model = VGG16_DWC(n_classes=test_opts.n_classes, input_ch=test_opts.input_ch, pruning_ratio=test_opts.pruning_ratio, relative=True).to(device)

    pruned_model = VGG5_DWC(n_classes=test_opts.n_classes, input_ch=test_opts.input_ch, pruning_ratio=test_opts.pruning_ratio, relative=True).to(device)
    test(test_opts.epoch, pruned_model, vis, test_loader, criterion, test_opts, load=True)