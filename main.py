import os
from collections import Counter

import torch
import torch.nn as nn
# from torch.optim import lr_scheduler
from model import vgg1d_bn

from preprocess import NCLASSES
from loader import NMFCC
from loader import get_dataloaders


LR = 6e-4
CHECKPOINT_DIR = './checkpoints'


def train(model, optimizer, criterion, evaluator, train_loader,
          save_path, use_cuda, max_epochs=None, max_overfit=3):

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
            best_checkpoint = save_path + f'_e{epoch}_s{score:.3f}.pkl'
            print('Saving trained model...', end='')
            torch.save(model.state_dict(), best_checkpoint)
            print('finished!')

    print(f'Loading best model: {best_checkpoint}...', end='')
    model.load_state_dict(torch.load(best_checkpoint))
    print('finished!')


def cal_accuracy(model, dataloader):
    correct = 0
    total = 0
    cnt = Counter()
    for i, (data, target) in enumerate(dataloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(1)
        # print(pred.tolist())
        cnt.update(pred.tolist())
        correct += pred.eq(torch.squeeze(target)).sum().cpu()
        total += len(pred)
    acc = int(correct) / int(total)
    assert int(total) == len(dataloader.dataset)
    print(f'Accuracy: {acc}({correct}/{total})')
    print(cnt)
    return acc


if __name__ == '__main__':

    ntest = 6
    save_path = os.path.join(CHECKPOINT_DIR, f'lr{LR}_ntest{ntest}')

    train_loader, dev_loader, test_loader = get_dataloaders(ntest=ntest)

    model = vgg1d_bn(in_channels=NMFCC, num_classes=NCLASSES)
    # model = vgg1d_bn(in_channels=NPERSEG//2+1, num_classes=NCLASSES)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    def evaluate_model(m):
        m.eval()
        dev_score = cal_accuracy(m, dev_loader)
        test_score = cal_accuracy(m, test_loader)
        print(f'dev score [{dev_score}], test score [{test_score}]')
        return dev_score

    try:
        train(model, optimizer=optimizer, criterion=criterion,
              evaluator=evaluate_model,
              train_loader=train_loader,
              save_path=save_path, use_cuda=use_cuda)
    except KeyboardInterrupt:
        pass

    evaluate_model(model)
