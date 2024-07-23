import os
from PIL import Image
from resnet18_bin import BinConv2d, resnet18_preact_bin 
import numpy as np
import torch 
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from utils import imshow 


import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="XNORNET-testing",
    name="#21_base_Resnet18",
    entity="scalar_go",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "XNOR-Net++",
    "dataset": "CIFAR10",
    "epochs": 80,
    }
)



hyper_param_epoch = 80
hyper_param_batch = 64
hyper_param_learning_rate = 0.001

#Dataset and Dataloader
transforms_train=transforms.Compose([transforms.Resize((256,256)), 
                                     transforms.CenterCrop((224,224)), #중앙 Crop
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor(), 
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])
transforms_test=transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])


train_data_set=torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms_train)
# train_data_set=torchvision.datasets.ImageFolder(root="./archive/train",transform=transforms_train)
train_loader=DataLoader(train_data_set,batch_size=hyper_param_batch,shuffle=True) #배치 단위로 loader

test_data_set= torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms_test)
# test_data_set=torchvision.datasets.ImageFolder(root="./archive/test",transform=transforms_train)
test_loader=DataLoader(test_data_set,batch_size=hyper_param_batch,shuffle=False)


#cifar10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if not (train_data_set.classes ==test_data_set.classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model, Loss, Optimizer
net = resnet18_preact_bin(num_classes=10, output_height=224, output_width=224, binarize=False).to(device)#output_height=32 output_width=32
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=hyper_param_learning_rate, momentum=0.9, weight_decay=10e-6)



#학습 및 검증 결과를 저장할 리스트 초기화
train_losses=[]
train_accuracies=[]
test_losses=[]
test_accuracies=[]


#test와 train
for epoch in range(hyper_param_epoch):
    net.train()
    correct=0
    running_loss=0
    total=0
    
    for num , data in enumerate(train_loader):
        imgs,label =data # labels는 각 배치의 레이블을 담고 있는 1차원 텐서입니다. 
        imgs=imgs.to(device)
        label=label.to(device)
        
        optimizer.zero_grad()
        out=net(imgs)
        loss=criterion(out,label)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        _,predicted=out.max(1) #out의 크기는 (배치 크기, 클래스 수, 높이, 너비) => predicted: 클래스 인덱스
        total+=label.size(0)
        correct+=predicted.eq(label).sum().item()

    
    train_loss=running_loss/len(train_loader)
    train_accuracy=100*correct/total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
        
    # log metrics to wandb
    wandb.log({ "train loss": train_loss, "train acc":train_accuracy},step=epoch)

    
    net.eval()
    test_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for num,data in enumerate(test_loader):
            imgs, label = data
            imgs = imgs.to(device)
            label = label.to(device)

            outs=net(imgs)
            loss=criterion(outs,label)
            test_loss+=loss.item()
            _,predicted=outs.max(1)
            total+=label.size(0)
            correct+=predicted.eq(label).sum().item()

    test_loss=test_loss/len(test_loader)
    test_accuracy=100*correct/total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # log metrics to wandb
    wandb.log({ "test loss": test_loss,"test acc":test_accuracy},step=epoch)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%,Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
print('Finished Training')

wandb.finish()

# #결과 시각화
# epochs=range(1,51)
# plt.figure(figsize=(12,4))

# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_losses, label='Train Loss')
# plt.plot(epochs, test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_accuracies, label='Train Accuracy')
# plt.plot(epochs, test_accuracies, label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()

# wandb.finish()

