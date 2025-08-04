from datetime import datetime
import os
import random

import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from networks.supermasks import Projector

from networks.MINE import Mine_conv
import os.path as osp

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import argparse
import yaml
# from networks.clsformer import SegFormer
from networks.CWsegformer import SegFormer
import monai
from train_process import Trainer, TrainerDP, Trainerbase
from train_process import TrainerMI

from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from modeling.deeplab import DeepLab
from tqdm import tqdm 
from torch.optim import SGD

local_path = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-g', '--gpu', type=int, default=3, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    
    parser.add_argument('--datasetTrain', nargs='+', type=int, default=[1,2,3], help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=[4], help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training the model')
    parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
    parser.add_argument('--max-epoch', type=int, default=400, help='max epoch')
    parser.add_argument('--stop-epoch', type=int, default=200, help='stop epoch')
    parser.add_argument('--interval-validate', type=int, default=10, help='interval epoch number to valide the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate',)
    parser.add_argument('--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr')
    parser.add_argument('--lam', type=float, default=0.9, help='momentum of memory update',)
    parser.add_argument('--data-dir', default='./Dataset/Fundus/', help='data root path')

    args = parser.parse_args()

    now = datetime.now()
    
    args.out = osp.join(local_path, 'logs', 'Unet','3',now.strftime('%Y%m%d_%H%M%S.%f'))
    # args.out = osp.join(local_path, 'logs', 'train_MI_2', now.strftime('%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    torch.cuda.manual_seed(1337)
    #
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    np.random.seed(1337)
    random.seed(1337)
    # torch.backends.cudnn.deterministic = True

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(256),
        # tr.RandomScaleCrop(512),
        # tr.RandomCrop(512),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        # tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomCrop(256),
        # tr.RandomCrop(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = DL.FundusSegmentation(base_dir=args.data_dir, phase='train', splitid=args.datasetTrain,
                                                         transform=composed_transforms_tr)
    train_loader = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                       transform=composed_transforms_ts)
    val_loader = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SegFormer(num_classes=2, num_domain=3, phi='b2', pretrained=True) # seg

#     model = monai.networks.nets.UNet(
#     spatial_dims=2,  
#     in_channels=3,
#     out_channels=2,  
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=0,
# )

    # load weights
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    start_epoch = 0
    start_iteration = 0

    optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=1e-4)


    trainer = Trainer.Trainer(
        cuda=cuda,
        model=model,
        lr=args.lr,
        lr_decrease_rate=args.lr_decrease_rate,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()