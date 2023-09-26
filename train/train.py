import sys
sys.path.append('../../util')

import argparse
import os
import shutil
import time
import datetime
import numpy as np
from pprint import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision import transforms
from tensorboardX import SummaryWriter
import pandas as pd
from multiscaleloss import EPE

from harps_spec_info import harps_spec_info
from fits_spec_datalayer import fits_spec_datalayer
import models


#--- Solver settings
n_iter = 0
solver = 'adam'
lr = 0.0001
weight_decay = 0
bias_decay = 0
momentum = 0.9
beta = 0.999
milestones = []
batch_size = 32
val_batch_size = 32
workers = 8
epoch_size = 0 #dataset size
sparse = False
multiscale_weights = [1];

print_freq = 10
print_freq_val = 100
checkpoint_freq_iter = 50000
val_freq_iter = 5000
val_save_freq_iter = 0 #must be divisible by val_freq_iter; 0=disable

parser = argparse.ArgumentParser(description='PyTorch implementation of Stellar-Karaoke',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu', metavar='gpu', default=0, type=int)

def main():
    args = parser.parse_args()

    global save_path,n_iter,device
    save_path = './output'

    #-- handle resume case
    pretrained_model = None

    #-- take care of the output dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        sys.exit('Error: output folder already exists.')

    #- create writer objects
    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    val_writer = SummaryWriter(os.path.join(save_path,'val'))

    #--- Set the GPU number
    device = torch.device('cuda:%d'%args.gpu if args.gpu >= 0 else 'cpu')

    #--- Dataset and loader
    trainset_length = 265000
    dataset_train = fits_spec_datalayer('../data/harps/fits',harps_spec_info,median_norm=True,median_threshold=50,max_index=trainset_length-1)
    dataset_val = fits_spec_datalayer('../data/harps/fits',harps_spec_info,median_norm=True,median_threshold=50,min_index=trainset_length)

    #-- Weighted sampling
    from torch.utils.data import WeightedRandomSampler
    metadata = pd.read_csv('../data/harps/metadata_our_dataset_count.csv')
    samples_count = metadata['fixed_name_count'][:trainset_length]
    sample_weights = 1. / samples_count
    sampler = WeightedRandomSampler(sample_weights, trainset_length, replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size,
        num_workers=workers, worker_init_fn = loader_worker_init, persistent_workers=True,
        pin_memory=True, shuffle=False, drop_last=True,
        sampler=sampler)

    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=val_batch_size,
        num_workers=1, worker_init_fn = loader_worker_init, persistent_workers=True,
        pin_memory=False, shuffle=False, drop_last=True)

    #--- Create the model
    print("=> creating model...")
    model = models.ae1d(None).to(device)

    print(model)

    epoch_start = 0

    #--- Set up the solver
    cudnn.benchmark = True
    print('=> setting {} solver'.format(solver))
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': bias_decay},
                    {'params': model.weight_parameters(), 'weight_decay': weight_decay}]

    optimizer = torch.optim.Adam(param_groups, lr,
                                 betas=(momentum, beta))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5, last_epoch = epoch_start-1)

    # Training loop
    for epoch in range(epoch_start,1000):
        try:
            train_loss = train(train_loader, model, optimizer, epoch, train_writer, val_loader, val_writer)
            scheduler.step()

def train(loader, model, optimizer, epoch, writer, val_loader,val_writer):
    global n_iter,epoch_size,device,val_freq_iter

    batch_time = AverageMeter()
    data_time = AverageMeter()
    reconst_losses = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()

    epoch_size = len(loader) if epoch_size == 0 else min(len(loader), epoch_size)

    end = time.time()

    for i, spectra in enumerate(loader):

        # Run validation if necessary
        if(n_iter % val_freq_iter == 0 and n_iter > 0):
            validate(val_loader,model,val_writer)

        # switch to train mode
        model.train()

        # measure data loading time
        data_time.update(time.time() - end)


        input = spectra.clone().to(device)
        target = spectra.to(device)

        # Forward pass
        output,mu,logvar = model(input)

        # compute loss
        tiled_mask = torch.from_numpy(nima_spec_datalayer.get_artifact_mask(harps_spec_info,
                                                                            batch_size = output.shape[0])).to(device)
        reconst_loss = EPE(output, target, mean=True, tiled_mask = tiled_mask)
        KLD_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


        if (n_iter - 5000) / (20000 - 5000) < 0:
            wkl = 0
        elif (n_iter - 5000) / (20000 - 5000) > 1:
            wkl = 0.3
        else:
            wkl = 0.3 * (n_iter - 5000) / (20000 - 5000)
        loss = reconst_loss + wkl * KLD_loss

        # record loss
        reconst_losses.update(reconst_loss.item(), target[0].size(0))
        kld_losses.update(KLD_loss.item(), target[0].size(0))
        losses.update(loss.item(), target[0].size(0))
        writer.add_scalar('loss', loss.item(), n_iter)



        # backward+update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                  'Iter: {0} Epoch: [{1}][{2}/{3}]\t Time {4}\t Data {5}\t Loss {6}+{7}={8} LR {9}'
                  .format(n_iter,epoch, i, epoch_size, batch_time, data_time, reconst_losses, kld_losses, losses, optimizer.param_groups[1]['lr']))
            sys.stdout.flush()
            print('wkl:',wkl)

        n_iter += 1

        if(n_iter % checkpoint_freq_iter == 0):
            print('saving checkpoint')
            save_checkpoint({
                'iter': n_iter,
                'epoch': epoch + 1,
                'lr': optimizer.param_groups[1]['lr'],
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            })

        if i >= epoch_size:
            break

    return losses.avg

def validate(loader, model):
    global n_iter,device

    epoch_size = len(loader)

    with torch.no_grad():
        model.eval()

        for i, spectra in enumerate(loader):

            input = spectra.clone().to(device)
            target = spectra.to(device)

            # Forward pass
            output,mu,logvar = model(input)

            # compute loss
            tiled_mask = torch.from_numpy(nima_spec_datalayer.get_artifact_mask(harps_spec_info,
                                                                                batch_size = output.shape[0])).to(device)
            loss = EPE(output, target, mean=True, tiled_mask = tiled_mask)

            # record loss
            losses.update(loss.item(), target[0].size(0))

            # logging

    print(' * VALIDATION LOSS {:.3f}'.format(losses.avg))

    return losses.avg


def save_checkpoint(state, filename=None):
    if not filename:
        filename = 'checkpoint_epoch%07d_iter%07d.pth.tar'%(state['epoch'],state['iter'])

    torch.save(state, os.path.join(save_path,filename))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

if __name__ == '__main__':
    main()
