import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_load.contrastive_learning_dataset import ContrastiveLearningDataset   #用于加载对比学习的数据集
from models.resnet_simclr import ResNetSimCLR  #从model文件夹中导入resnet_simclr文件（模型框架）
from simclr import SimCLR  #导入simclr文件中的SimCLR类（训练框架）
import numpy as np
import logging
import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):   #实现SimCLR算法训练过程

    def __init__(self, *args, **kwargs):  #初始化模型、优化器、调度器、日志记录器和损失函数
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):  #交叉熵损失函数
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):  #训练函数，加载训练数据集、训练迭代

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in tqdm(range(self.args.epochs)):
            for images, _ in train_loader:
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:  
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training finish.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({  #每个 epoch 都对模型和优化器状态进行保存
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint saved at {self.writer.log_dir}.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',help='dataset name', choices=['stl10', 'cifar10'])

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  choices=model_names,
                    help='model architecture: ' +' | '.join(model_names) +' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)',dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int, help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=4, type=int, help='Gpu index.')


def main():
    args = parser.parse_args()  #解析参数并存储在args
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # gpu加速
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    #加载数据集
    dataset = ContrastiveLearningDataset(args.data)
    
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    subset_indices = np.random.choice(len(train_dataset), size=int(len(train_dataset)), replace=False) 

    #创建训练集加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)  #初始化，model-resnet_simclr

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)  #优化器

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)  #学习率调度器

    with torch.cuda.device(args.gpu_index):  #在训练集上用gpu训练resnet18
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
