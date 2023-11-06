import torch.optim as optim
from torch import nn
from tqdm import tqdm
import utils
import torch

def train(net, trainloader, epochs, device):
    print("train phase")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=utils.lr, weight_decay=utils.weight_decay, nesterov=True, momentum=utils.momentum)

    for epoch in range(epochs):  # 데이터셋 2번 받기
        adjust_learning_rate(optimizer, epoch)
        net.train()
        running_loss = 0.0
        bar = tqdm(trainloader, unit="batch", desc=f"Epoch {epoch + 1}", ncols=70)
        for data in bar:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # inputs = torch.autograd.Vari able(labels)
            # 학습
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            bar.set_postfix(loss=loss.item())
    print()
    print('Finished Training')
    return net


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = utils.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr