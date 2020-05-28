# train and save model
from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar as Bar
import numpy as np
from dataset import *
from model import *
from utils import *



parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='dsprites', type=str)
parser.add_argument( '--loss', default='bernoulli', type=str)
parser.add_argument( '--interval', default=1000, type=int)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epoch', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--C_max', default=25.0, type=float)
parser.add_argument('--C_stop_iter', default=20000, type=int)
parser.add_argument('--C_step', default=1000, type=int)
parser.add_argument('--C_start', default=0.5, type=float)
parser.add_argument('--hinge_loss', default=0, type=int)
parser.add_argument('--original', default=0, type=int)
parser.add_argument('--gamma', type=float, default=1000)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                    metavar='W')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='test/', type=str)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()


args.save_dir="results/"+args.save_dir
use_cuda = torch.cuda.is_available()

# Random seed
# if args.manualSeed is None:
#     args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)




def main():
    # add summery writer and working path
    writer = SummaryWriter("{}/tblogs/train".format(args.save_dir))
    state={}
    # Data TODO: load dataloader dataloader,testdataloader
    state["dataloader"],state["testdataloader"]=get_dataloader(args)
    # print(len(state["testdataloader"]))
    # Model TODO: load model"""
    state["model"]=get_model(args,use_cuda)
    cudnn.benchmark = True
    state["criterion"] = get_lossfuc(args)
    
    optimizer = optim.Adam(state["model"].parameters(), lr=args.lr, betas=(0.9,0.999),weight_decay=args.weight_decay,amsgrad=True)

    state["optimizer"]=optimizer
    state["writer"]=writer
    state["use_cuda"]=use_cuda
    state["args"]=args
    state["epoch"]=args.epoch
    state["total_iterations"]=state["epoch"]*len(state["dataloader"])
    print("Start training, there are total {} iterations".format(state["total_iterations"]))
    train(state)

    writer.close()
    save_checkpoint(state["model"],args.save_dir,args.epoch)

def train(state):
    dataloader=state["dataloader"]
    model=state["model"]
    criterion=state["criterion"]
    optimizer=state["optimizer"]
    writer=state["writer"]
    use_cuda=state["use_cuda"]
    args=state["args"]
    epoch=state["epoch"]
    iteration=0
    print(args)
    C=get_C(iteration,state,args)

    # switch to train mode
    model.train()
    test_batch=None
    if state["testdataloader"] is not None:
        test_batch=next(iter(state["testdataloader"]))[0].cuda()
        print("the len of test set is {}".format(test_batch.shape[0]))

    for epoch in range(state["epoch"]):
        print("This is epoch {}".format(epoch))
        # continue

        batch_time = AverageMeter()
        data_time = AverageMeter()
        recon_losses = AverageMeter()
        KLD_losses = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=int(len(dataloader)/args.interval)+1 )

        
        for batch_idx, inputs in enumerate(dataloader):
            inputs=torch.tensor(inputs[0])
            # if args.dataset!="dsprites":
            #     inputs=inputs.float()
            iteration+=1
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs= inputs.cuda()
            
            if test_batch is None:
                test_batch=inputs.clone()
            # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs, mean, logvar = model(inputs)
            recon_loss = criterion(outputs, inputs,reduction='sum')
            KLD_loss,KLD_loss_dim=model.KL_divergence(mean,logvar)
            if iteration%args.C_step==0:
                C=get_C(iteration,state,args)
            if args.hinge_loss:
                loss=recon_loss+args.gamma*F.relu(KLD_loss-C)
            elif args.original :
                loss=recon_loss+args.gamma*KLD_loss
            else:
                loss=recon_loss+args.gamma*(KLD_loss-C).abs()

            recon_losses.update(recon_loss.data.item(), inputs.size(0))
            KLD_losses.update(KLD_loss.data.item(), inputs.size(0))
            losses.update(loss.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # tensorboard
            if iteration%args.interval==1:
                if iteration%(args.interval*50)==1:
                    writer.add_histogram('mean', mean.cpu(), iteration)
                    writer.add_histogram('logvar', logvar.cpu(), iteration)
                    test(test_batch=test_batch, model=model,state=state,step=iteration)
                writer.add_scalar('abs_mean', mean.cpu().abs().mean(), iteration)
                writer.add_scalar('abs_logvar', logvar.cpu().abs().mean(), iteration)
                writer.add_scalar('recon_loss', recon_loss.data.item(), iteration)
                writer.add_scalar('KLD_loss', KLD_loss.data.item(), iteration)
                writer.add_scalar('loss', loss.data.item(), iteration)
                for dim in range(len(KLD_loss_dim)):
                    writer.add_scalar('KLD_loss_dim_{}'.format(dim), KLD_loss_dim[dim], iteration)

                pass

                # plot progress

                bar.suffix  = 'Epoch {epoch} ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | recon_loss: {recon_loss: .4f} | KLD_loss: {KLD_loss: .4f} | C:{C: .4f}'.format(
                            epoch=epoch,
                            batch=batch_idx + 1,
                            size=len(state["dataloader"]),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            recon_loss=recon_losses.avg,
                            KLD_loss=KLD_losses.avg,
                            C=C
                            )
                bar.next()
        bar.finish()
    test(test_batch=test_batch, model=model,state=state,step=iteration)
    return (recon_losses.avg, KLD_losses.avg,losses.avg)

def test(test_batch, model,state,step):
    # switch to evaluate mode
    model.eval()

    inputs=test_batch
    if step==1:
        if use_cuda:
            state["writer"].add_image("input", get_image([inputs.cpu().numpy(),]), global_step=step,dataformats='HWC')
            inputs= inputs.cuda()
        else:
            state["writer"].add_image("input", get_image([inputs.numpy(),]), global_step=step,dataformats='HWC')

    # compute output of mean
    with torch.no_grad():
        outputs, mean = model.forward_with_mean(inputs,loss=args.loss)
        if use_cuda:
            outputs= outputs.cpu()
    state["writer"].add_image("output of mean", get_image([outputs.numpy(),]), global_step=step,dataformats='HWC')

    # forward_explore_var
    for picture in [0,1,4,10,20,48,35,12,34,16,15,27,35,29,42]:
        outputs=[]
        with torch.no_grad():
            for i in range(model.z_size):
                output = model.forward_explore_var(inputs[picture,:,:,:].unsqueeze(0),axis=i,loss=args.loss)
                if use_cuda:
                    output= output.cpu()
                outputs.append(output.numpy())
        img=get_image(outputs)
        state["writer"].add_image("explore var {}".format(picture), img, global_step=step,dataformats='HWC')


    model.train()
    return 


if __name__ == '__main__':
    main()
