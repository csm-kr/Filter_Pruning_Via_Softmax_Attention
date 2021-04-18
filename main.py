import os
import sys
import torch
from torch.utils.data import DataLoader
import visdom
from model import build_model
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from train import train
from test import test
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from config import device
from dataset import build_dataset
from config import parser

from pruner import prune


def main():
    opts = parser(sys.argv[1:])
    train_set, test_set = build_dataset(opts)

    # visdom
    vis = visdom.Visdom()

    # data loader
    train_loader = DataLoader(dataset=train_set,
                              shuffle=True,
                              batch_size=opts.batch_size)

    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=256)

    # model
    model, pruned_model = build_model(opts)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(params=model.parameters(),
                          lr=opts.lr,
                          weight_decay=1e-4,
                          momentum=0.9)

    # 9. scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[opts.milestone_0, opts.milestone_1], gamma=0.1)
    best_acc = 0

    ###################################################
    #             training and pruning
    ###################################################
    print("training and pruing...")
    for epoch in range(opts.epoch):

        train(epoch, model, vis, train_loader, criterion, optimizer, opts)
        accuracy = test(epoch, model, vis, test_loader, criterion, opts)
        if best_acc < accuracy:
            best_acc = accuracy
            print("best_accuracy : *** {:.4f}% *** (improved)".format(best_acc * 100.))

            # 13. pruning
            prune(model, pruned_model, opts.pruning_ratio)
            if not os.path.exists(opts.save_path):
                os.mkdir(opts.save_path)
            torch.save(pruned_model.state_dict(),
                       os.path.join(opts.save_path, '{}_{}_{}_{}_{:.4f}.pth.tar'.format(opts.pruned_save_file_name, opts.model_name, opts.dataset_type, opts.pruning_ratio, best_acc)))

        elif best_acc >= accuracy:

            print("current_accuracy : {:.4f}%".format(accuracy * 100.))
            print("best_accuracy : *** {:.4f}% *** (downgrade)".format(best_acc * 100.))

        scheduler.step()

    ###################################################
    #                     fine tune
    ###################################################

    # load best acc models
    pruned_model.load_state_dict(torch.load('./saves/{}_{}_{}_{}_{:.4f}.pth.tar'.format(opts.pruned_save_file_name, opts.model_name, opts.dataset_type, opts.pruning_ratio, best_acc)))

    # optimizer
    optimizer = optim.SGD(params=pruned_model.parameters(),
                          lr=opts.lr,
                          weight_decay=1e-4,
                          momentum=0.9)
    # scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[100, 150], gamma=0.1)
    best_acc = 0

    for epoch in range(opts.epoch):
        print("finetuning...")

        train(epoch, model, vis, train_loader, criterion, optimizer, opts)
        accuracy = test(epoch, pruned_model, vis, test_loader, criterion, opts)

        if best_acc < accuracy:
            best_acc = accuracy
            print("best_accuracy : *** {:.4f}% *** (improved)".format(best_acc * 100.))
            if not os.path.exists(opts.save_path):
                os.mkdir(opts.save_path)
            torch.save(pruned_model.state_dict(),
                       os.path.join(opts.save_path, '{}_{}_{}_{}_{:.4f}.pth.tar'.format(opts.save_file_name, opts.model_name, opts.dataset_type, opts.pruning_ratio, best_acc)))

        elif best_acc >= accuracy:

            print("current_accuracy : {:.4f}%".format(accuracy * 100.))
            print("best_accuracy : *** {:.4f}% *** (downgrade)".format(best_acc * 100.))

        scheduler.step()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    print("empty_cuda_cache")
    main()
