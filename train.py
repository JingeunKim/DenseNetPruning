import torch.optim as optim
from torch import nn
from tqdm import tqdm
import utils
import time
import test
# def train(train_loader, model, criterion, optimizer, epoch):
def train(net, trainloader, epochs, device, testloader):
    """Train for one epoch on the training set"""
    print("train phase")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=utils.lr, weight_decay=utils.weight_decay, nesterov=True, momentum=utils.momentum)
    best_error = 100
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)
        net.train()
        running_loss = 0.0
        bar = tqdm(trainloader, unit="batch", desc=f"Epoch {epoch + 1}", ncols=70)
        for data in bar:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=loss.item())
        # print("running_loss = ", loss.item())
        error_rate = test.test(testloader, net, utils.device)
        if best_error > error_rate:
            best_error = error_rate
        print('best_error :', best_error)
    print()
    print('Finished Training')
    return net, best_error

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = utils.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    # if args.tensorboard:
    # print('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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




def GAtrain(model, train_loader, epochs, device):
    print("GAtrain")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=utils.lr, weight_decay=utils.weight_decay, nesterov=True,
                          momentum=utils.momentum)
    # switch to train mode
    model.train()
    for epoch in range(epochs):
        # end = time.time()
        for i, (input, target) in enumerate(train_loader):
            target = target.to(device)
            input = input.to(device)

            output = model(input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('Finished Training')
    print("loss = ", loss.item())
    return model, loss.item()