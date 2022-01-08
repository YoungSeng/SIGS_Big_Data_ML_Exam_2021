import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import torch.optim
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset

from dataset import Food_LT
# from model import resnet34
from model import ResNet50
# from model import ResNet101
# from model import ResNet152
# from DenseNet import densenet121
# from DenseNet import densenet161
# from EfficientNet import efficientnet_b0
# from EfficientNet import efficientnet_b1
# from Regnet import regnet_x_1_6gf
from Regnet import regnet_x_3_2gf
import config as cfg
from utils import adjust_learning_rate, save_checkpoint, train, validate, logger



# class EMA():
#     def __init__(self, model, decay):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}
#
#     def register(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()
#
#     def update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 new_average = (1.0 - self.decay) * param.data.cpu() + self.decay * self.shadow[name]
#                 self.shadow[name] = new_average.clone()
#
#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 self.backup[name] = param.data
#                 param.data = self.shadow[name].cuda()
#
#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name].cuda()
#         self.backup = {}


def main():
    # model = resnet34()
    # model = ResNet50()
    # model = ResNet101()
    # model = ResNet152()
    # model = densenet121()
    # model = densenet161()
    # model = efficientnet_b0()
    # model = efficientnet_b1()
    model = regnet_x_3_2gf()

    # ema = EMA(model, 0.999)
    # ema.register()

    if cfg.resume:
        ''' plz implement the resume code by ur self! '''
        pass

    print('log save at:' + cfg.log_path)
    logger('', init=True)
    
    if not torch.cuda.is_available():
        logger('Plz train on cuda !')
        os._exit(0)

    if cfg.gpu is not None:
        print('Use cuda !')
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        # model.to('cuda')  # 先把模型放到当前进程的GPU中去
        # print(range(torch.cuda.device_count()))
        # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    print('Load dataset ...')
    dataset = Food_LT(False, root=cfg.root, batch_size=cfg.batch_size, num_works=4)

    train_loader = dataset.train_instance
    val_loader = dataset.eval
    
    criterion = nn.CrossEntropyLoss().cuda(cfg.gpu)
    optimizer = torch.optim.SGD([{"params": model.parameters()}], cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    # optimizer = torch.optim.Adam([{"params": model.parameters()}])
    
    best_acc = 0
    for epoch in range(cfg.num_epochs):
        logger('--'*10 + f'epoch: {epoch}' + '--'*10)
        logger('Training start ...')
        
        adjust_learning_rate(optimizer, epoch, cfg)
        
        train(train_loader, model, criterion, optimizer, epoch)
        logger('Wait for validation ...')
        acc = validate(val_loader, model, criterion)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        logger('* Best Prec@1: %.3f%%' % (best_acc))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_model': model.state_dict(),
            'best_acc': best_acc,
        }, is_best, cfg.root)



    print('Finish !')


if __name__ == '__main__':
    main()