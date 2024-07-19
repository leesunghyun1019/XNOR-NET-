import os
from PIL import Image
from resnet18_bin import BinConv2d, resnet18_preact_bin 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt

hyper_param_epoch = 10
hyper_param_batch = 200
hyper_param_learning_rate = 0.001

# Dataset and Dataloader
transforms_train = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data_set = torchvision.datasets.ImageFolder(root="./dataset/train", transform=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = torchvision.datasets.ImageFolder(root="./dataset/test", transform=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=False)

if not (train_data_set.classes == test_data_set.classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model, Loss, Optimizer
net = resnet18_preact_bin(num_classes=6, output_height=224, output_width=224, binarize=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=hyper_param_learning_rate, momentum=0.9, weight_decay=10e-6)

total_batch = len(train_loader)

#test
net.train()
epoch_loss_history = []
for epoch in range(hyper_param_epoch):
    avg_cost = 0.0
    for num, data in enumerate(train_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = net(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        avg_cost += loss.item() / total_batch
    epoch_loss_history.append(avg_cost)
    print('[Epoch:{}] cost={:.6f}'.format(epoch + 1, avg_cost))
print('Learning Finished!')

# 손실 값 그래프 그리기
plt.plot(range(1, hyper_param_epoch + 1), epoch_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#train
net.eval()
total_accuracy = 0.0
total_samples = 0

with torch.no_grad():
    for num, data in enumerate(test_loader):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)
        
        prediction = net(imgs)
        correct_prediction = torch.argmax(prediction, 1) == label
        accuracy = correct_prediction.float().mean()
        
        total_accuracy += accuracy.item() * imgs.size(0)
        total_samples += imgs.size(0)
        
        print('Batch Accuracy:', accuracy.item())
       

overall_accuracy = total_accuracy / total_samples
print('Overall Accuracy:', overall_accuracy)
