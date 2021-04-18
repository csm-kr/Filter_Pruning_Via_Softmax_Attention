import os
import time
import torch
from config import device


def train(epoch, model, vis, data_loader, criterion, optimizer, vis_name='training_loss'):

    model.train()
    tic = time.time()
    # 11. train
    for idx, (img, target) in enumerate(data_loader):

        img = img.to(device)  # [N, 1, 28, 28]
        target = target.to(device)  # [N]
        output = model(img)  # [N, 47]
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        if idx % 10 == 0:
            vis.line(X=torch.ones((1, 1)) * idx + epoch * len(data_loader),
                     Y=torch.Tensor([loss]).unsqueeze(0),
                     update='append',
                     win=vis_name,
                     opts=dict(x_label='step',
                               y_label='loss',
                               title='loss',
                               legend=['total_loss']))

            print('Epoch : {}\t'
                  'step : [{}/{}]\t'
                  'loss : {}\t'
                  'lr   : {}\t'
                  'time   {}\t'
                  .format(epoch,
                          idx, len(data_loader),
                          loss,
                          lr,
                          time.time() - tic))
