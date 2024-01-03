import torch
from tqdm import tqdm
def test(testloader, model, device):
    correct = 0
    total = 0
    print("test")
    model.eval()
    incorrect=0
    with torch.no_grad():
        for data in tqdm(testloader, unit="batch"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # correct += (predicted == labels).float().sum().item()
            correct += predicted.eq(labels).sum().item()
            incorrect += predicted.ne(labels.data).cpu().sum()

            error_rate = 100. * incorrect / total

            # error_rate = 100 - acc
    print('error_rate of the network on the 10000 test images: {}'.format(error_rate))
    # all_loss.append(error_rate)
    return error_rate

# def test(val_loader, model, device):
#     """Perform validation on the validation set"""
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#
#     criterion = nn.CrossEntropyLoss()
#     # switch to evaluate mode
#     model.eval()
#
#     end = time.time()
#     for i, (input, target) in enumerate(val_loader):
#         target = target.to(device)
#         input = input.to(device)
#
#         # compute output
#         output = model(input)
#         # loss = criterion(output, target)
#
#         # measure accuracy and record loss
#         prec1 = accuracy(output.data, target, topk=(1,))[0]
#         # losses.update(loss.data, input.size(0))
#         top1.update(prec1, input.size(0))
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % 10 == 0:
#             print('Test: [{0}/{1}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                 i, len(val_loader), batch_time=batch_time,
#                 top1=top1))
#
#     print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
#     # log to TensorBoard
#     # if args.tensorboard:
#     #     log_value('val_loss', losses.avg, epoch)
#     #     log_value('val_acc', top1.avg, epoch)
#     return top1.avg


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
