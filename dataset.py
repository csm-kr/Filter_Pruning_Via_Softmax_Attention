import torchvision.transforms as tfs
from torchvision.datasets import EMNIST, KMNIST, CIFAR100, MNIST, FashionMNIST, QMNIST, CIFAR10


def build_dataset(opts):
    """
    funtion to make datasets and transforms
    :param dataset_name:
    :return:
    """
    assert opts.dataset_type in ['mnist', 'fashionmnist', 'cifar10', ]

    train_set = None
    test_set = None

    transform_mnist = tfs.Compose([tfs.RandomPerspective(),
                                   tfs.RandomRotation(10, fill=(0,)),
                                   tfs.ToTensor(),
                                   tfs.Normalize((0.1307,), (0.3081,))
                                   ])

    test_transfrom_mnist = tfs.Compose([tfs.ToTensor(),
                                        tfs.Normalize((0.1307,), (0.3081,))
                                        ])

    transform_cifar = tfs.Compose([
        tfs.Pad(4),
        tfs.RandomCrop(32),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2023, 0.1994, 0.2010)),
    ])

    test_transform_cifar = tfs.Compose([tfs.ToTensor(),
                                        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                      std=(0.2023, 0.1994, 0.2010)),
                                        ])
    if opts.dataset_type == 'mnist':

        train_set = MNIST('./data/MNIST',
                          train=True,
                          download=True,
                          transform=transform_mnist)

        test_set = MNIST('./data/MNIST',
                         train=False,
                         download=True,
                         transform=test_transfrom_mnist)

        opts.n_classes = 10
        opts.input_ch = 1

    elif opts.dataset_type == 'fashionmnist':

        train_set = FashionMNIST('./data/FashionMNIST',
                                 train=True,
                                 download=True,
                                 transform=transform_mnist)

        test_set = FashionMNIST('./data/FashionMNIST',
                                train=False,
                                download=True,
                                transform=test_transfrom_mnist)

        opts.n_classes = 10
        opts.input_ch = 1

    elif opts.dataset_type == 'cifar10':

        train_set = CIFAR10('./data/CIFAR10',
                            train=True,
                            download=True,
                            transform=transform_cifar)

        test_set = CIFAR10('./data/CIFAR10',
                           train=False,
                           download=True,
                           transform=test_transform_cifar)

        opts.n_classes = 10
        opts.input_ch = 3

    return train_set, test_set



