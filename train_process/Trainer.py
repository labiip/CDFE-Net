from datetime import datetime
from math import inf
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import early_stopping
from networks.MINE import Mine_conv, kl_divergence, mutual_info_kl_loss
import os.path as osp
import timeit
from torchvision.utils import make_grid
import numpy as np
import torch
import torch.nn.functional as F
import pytz
from tensorboardX import SummaryWriter

import tqdm
import socket
from utils.metrics import *
from utils.Utils import *
from networks.supermasks import mask_overlap

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()
softmax = torch.nn.Softmax(-1)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class Trainer(object):

    def __init__(self, cuda, model, lr, val_loader, train_loader, out, max_epoch, optim, stop_epoch=None,
                 lr_decrease_rate=0.1, interval_validate=None, batch_size=1):
        self.cuda = cuda
        self.model = model
        self.optim = optim
        self.lr = lr
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.train_loader = train_loader
        self.time_zone = 'Asia/Shanghai'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/cup_dice',
            'train/disc_dice',
            'valid/loss_CE',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.acc = 0.0
        self.best_val = inf
        self.best_epoch = -1
        

 
    def validate(self):
        training = self.model.training
        self.model.eval()

        val_loss = 0
        val_cup_dice = 0
        val_disc_dice = 0
        metrics = []
        loss_cls = 0
        
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):

                image = sample['image']
                label = sample['label']
                domain_code = sample['dc']

                data = image.cuda()
                target_map = label.cuda()
                domain_code = domain_code.cuda()

                with torch.no_grad():
        
                    _, predictions, domain_predict, mask, re, ir,_= self.model(data)

                loss_seg = bceloss(torch.sigmoid(predictions), target_map)
                loss_cls = mseloss(softmax(domain_predict), domain_code)
                loss = loss_seg
           
                val_loss += loss

                dice_cup, dice_disc = dice_coeff_2label(np.asarray(torch.sigmoid(predictions.data.cpu())) > 0.75, target_map)
    
                val_cup_dice += dice_cup
                val_disc_dice += dice_disc
            
            val_cup_dice /= len(self.val_loader)
            val_disc_dice /= len(self.val_loader)
            metrics.append((val_loss, val_cup_dice, val_disc_dice))
            self.writer.add_scalar('val_data/loss', val_loss, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.train_loader)))

            # print('\n Average valLoss: %f' % (val_loss)) # cls

            mean_dice = val_cup_dice + val_disc_dice
            is_best = mean_dice > self.best_mean_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model.__class__.__name__,
                    'optim_state_dict': self.optim.state_dict(),
                    'model_state_dict': self.model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_best.pth.tar'))
            else:
                if (self.epoch + 1) % 20 == 0:
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': self.model.state_dict(),
                        'learning_rate_gen': get_lr(self.optim),
                        'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))

            if training:
                self.model.train()

    def train_epoch(self):
        self.model.train()
        self.running_seg_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0
        self.running_cls_loss = 0

        start_time = timeit.default_timer()
        for batch_idx, sample in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration

            assert self.model.training
            self.optim.zero_grad()
        

            image = None
            label = None
            domain_code = None
            
            for domain in sample:
                if image is None:
                    image = domain['image']
                    label = domain['label']
                    domain_code = domain['dc']
                else:
                    image = torch.cat([image, domain['image']], 0)
                    label = torch.cat([label, domain['label']], 0)
                    domain_code = torch.cat([domain_code, domain['dc']], 0)

            image = image.cuda()
            target_map = label.cuda()
            domain_code = domain_code.cuda()
            self.model = self.model.cuda()
            self.projector = self.projector.cuda()
            
            x_, pred, domain_predict, mask, re, ir, re_p = self.model(image) ######ours
            
            loss_seg = 2*bceloss(torch.sigmoid(pred), target_map)
            loss_x = bceloss(torch.sigmoid(x_), target_map)
            loss_re_p = bceloss(torch.sigmoid(re_p), target_map)
            
            loss_cls = mseloss(softmax(domain_predict), domain_code)
         
            kl = 0.02*mutual_info_kl_loss(re, ir)

            #best loss
            loss = loss_cls + loss_seg + loss_x + kl + loss_re_p
    
           
            self.running_seg_loss += loss
        
            loss.backward()
            self.optim.step()
            
            # write image log
            if iteration % 30 == 0:
                grid_image = make_grid(
                    image[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/image', grid_image, iteration)
                grid_image = make_grid(
                    target_map[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/target_cup', grid_image, iteration)
                grid_image = make_grid(
                    target_map[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/target_disc', grid_image, iteration)
                
                grid_image = make_grid(torch.sigmoid(pred)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/prediction_cup', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(pred)[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/prediction_disc', grid_image, iteration)

            # write loss log
            self.writer.add_scalar('train_gen/loss', loss, iteration)

        self.running_seg_loss /= len(self.train_loader)
        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f, Average Loss: %f, Execution time: %.5f' %
              (self.epoch, get_lr(self.optim), self.running_seg_loss, stop_time - start_time)) # cls

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch + 1) % (self.max_epoch//2) == 0:
                _lr_gen = self.lr * self.lr_decrease_rate
                for param_group in self.optim.param_groups:
                    param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr', get_lr(self.optim), self.epoch * (len(self.train_loader)))
            if (self.epoch + 1) % self.interval_validate == 0 or self.epoch == 0:
                self.validate()
        self.writer.close()



