import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
# from torchvision.models import vgg11_bn
from model import vgg11_bn

from loader import train_loader, dev_loader, test_loader
from preprocess import NCLASSES

LR = 0.00005
GAMMA = 0.3

CHECKPOINT_DIR = './checkpoints'

def train(model, optimizer, criterion, evaluator, train_loader, save_dir, use_cuda,
          max_epochs=None, max_overfit=10):

    print(f'Learning rate is: {optimizer.param_groups[0]["lr"]}')
    best_score = 0
    noverfits = 0
    epoch = 0

    while (max_epochs is None and noverfits < max_overfit) or (epoch < max_epochs):
        epoch += 1
        model.train()
        for i, (data, target) in enumerate(train_loader):
            target = torch.squeeze(target)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if i % 5 == 0:
                print(f'Epoch [{epoch}], iteration [{i+1}], loss [{loss}]')

        model.eval()
        score = evaluator(model)
        print(f'Epoch [{epoch}], score [{score}]')
        if score < best_score:
            noverfits += 1
        else:
            best_score = best_score
            print(f'Saving trained model after epoch {epoch}')
            filename = f'{model.name}_{epoch}.pkl'
            torch.save(model.state_dict(), os.path.join(save_dir, filename))

def cal_accuracy(model, dataloader):
    correct = 0
    for i, (data, target) in enumerate(dataloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(1)
        correct += pred.eq(target).sum().cpu()
    return correct / len(dataloader.dataset)

if __name__ == '__main__':

    model = vgg11_bn(num_classes=NCLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    use_cuda = torch.cuda.is_available()
    if use_cuda: model.cuda()

    train(model, optimizer=optimizer, criterion=criterion,
          evaluator=lambda m: cal_accuracy(m, dev_loader), train_loader=train_loader,
          save_dir=CHECKPOINT_DIR, use_cuda=use_cuda)

    test_acc = cal_accuracy(model, test_loader)
    print(f'Score on test set [{test_acc}]')
