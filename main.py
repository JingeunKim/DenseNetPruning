import shutil

import utils
import dataloader
import torch
from GA import GA
import datetime
from utils import logger
import time
import os
import torch.nn as nn
import torch.optim as optim

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/{:%Y%m%d}_{}/".format(datetime.datetime.now(), utils.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (str(datetime.datetime.now()) + utils.dataset) + 'model_best.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch, device):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('train_loss', losses.avg, epoch)
    #     log_value('train_acc', top1.avg, epoch)

def validate(val_loader, model, criterion, device):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('val_loss', losses.avg, epoch)
    #     log_value('val_acc', top1.avg, epoch)
    return top1.avg

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = utils.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# start = time.time()
# utils.print_and_log(logger, "Data augmentation : {}".format(utils.augmentation))
#
# GA(nDenseBlock=100, Bottleneck=True).evolve()
# end = time.time()
# utils.print_and_log(logger, "GA TIME : {}".format(end-start))

# model = torch.load('./models/model{:%Y%m%d}_{}_{}.pt'.format(datetime.datetime.now(), utils.prob, str(utils.augmentation)))
model = torch.load('./models/model20231227_0.5_True.pt')

utils.print_and_log(logger, "Model : model{:%Y%m%d}_{}_{}.pt".format(datetime.datetime.now(), utils.prob, str(utils.augmentation)))
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
utils.print_and_log(logger, "train & test")
utils.print_and_log(logger, "Best model # params =  {}".format(num_params))
trainloader, testloader, classes = dataloader.dataloader()
# best_prec1=0
# for epoch in range(utils.epochs):
#     # adjust_learning_rate(optimizer, epoch)
#     # train for one epoch
#     model = train(model, trainloader, utils.epochs, utils.device)
#     # evaluate on validation set
#     acc = test(testloader, model, utils.device)
#
#     # remember best prec@1 and save checkpoint
#     is_best = acc > best_prec1
#     best_prec1 = max(acc, best_prec1)
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'best_prec1': best_prec1,
#     }, is_best)
# print('Best accuracy: ', best_prec1)
# model = train(model, trainloader, utils.epochs, utils.device)
# acc = test(testloader, model, utils.device)

best_prec1 = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=utils.lr, weight_decay=utils.weight_decay, nesterov=True,
                      momentum=utils.momentum)
for epoch in range(utils.epochs):
    adjust_learning_rate(optimizer, epoch)
    # train for one epoch
    train(trainloader, model, criterion, optimizer, epoch, utils.device)

    # evaluate on validation set
    prec1 = validate(testloader, model, criterion, utils.device)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best)
print('Best accuracy: ', best_prec1)




utils.print_and_log(logger, "Best model error rate =  {}%".format(best_prec1))
utils.print_and_log(logger, "END")
