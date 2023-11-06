import torch
from tqdm import tqdm

def test(model, testloader, device):
    # all_loss = []
    correct = 0
    total = 0
    print("test")
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader, unit="batch"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).float().sum().item()
            error_rate = 100 * correct // total
            # error_rate = 100 - acc
    print('Accuracy of the network on the 10000 test images: {}'.format(error_rate))
    # all_loss.append(error_rate)
    return error_rate
