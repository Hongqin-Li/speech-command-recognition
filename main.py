import os

import torch
import torch.nn as nn
# from torch.optim import lr_scheduler
from model import vgg1d_bn

from preprocess import NCLASSES
from loader import BATCH_SIZE, NMFCC
from loader import get_dataloaders


LR = 4e-5
CHECKPOINT_DIR = './checkpoints'


def train(model, optimizer, criterion, evaluator, train_loader,
          save_dir, use_cuda, max_epochs=None, max_overfit=3):

    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    print(f'Train set length: {len(train_loader.dataset)}')

    best_score = 0
    best_checkpoint = ''
    noverfits = 0
    epoch = 0

    while (max_epochs is None and noverfits < max_overfit) or \
          (max_epochs is not None and epoch < max_epochs):
        epoch += 1

        model.train()
        for i, (data, target) in enumerate(train_loader):
            target = torch.squeeze(target)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch}], iteration [{i+1}], loss [{loss}]')

        model.eval()
        score = evaluator(model)
        print(f'Epoch [{epoch}], score [{score}]')
        if score < best_score:
            noverfits += 1
        else:
            noverfits = 0
            best_score = score
            filename = f'lr{LR}_bs{BATCH_SIZE}_e{epoch}_s{score:.2f}.pkl'
            best_checkpoint = os.path.join(save_dir, filename)
            print('Saving trained model...', end='')
            torch.save(model.state_dict(), best_checkpoint)
            print('finished!')

    print(f'Loading best model: {best_checkpoint}...', end='')
    model.load_state_dict(torch.load(best_checkpoint))
    print('finished!')


def cal_accuracy(model, dataloader):
    correct = 0
    total = 0
    for i, (data, target) in enumerate(dataloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(1)
        correct += pred.eq(torch.squeeze(target)).sum().cpu()
        total += len(pred)
    acc = int(correct) / int(total)
    assert int(total) == len(dataloader.dataset)
    print(f'Accuracy: {acc}({correct}/{total})')
    return acc


if __name__ == '__main__':

    train_loader, dev_loader, test_loader = get_dataloaders()

    model = vgg1d_bn(in_channels=NMFCC, num_classes=NCLASSES)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    try:
        train(model, optimizer=optimizer, criterion=criterion,
              evaluator=lambda m: cal_accuracy(m, dev_loader),
              train_loader=train_loader,
              save_dir=CHECKPOINT_DIR, use_cuda=use_cuda)
    except KeyboardInterrupt:
        pass

    model.eval()
    test_acc = cal_accuracy(model, test_loader)
    print(f'Score on test set [{test_acc}]')
