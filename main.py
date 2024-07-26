import os
from PIL import Image
from resnet18_bin import BinConv2d, resnet18_preact_bin
# from XNOR_Net import resnet18_XNOR
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

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="XNORNET-testing",
#     name="#25_XNORNET_Resnet18",
#     entity="scalar_go",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.001,
#     "architecture": "XNOR-Net++",
#     "dataset": "CIFAR10",
#     "epochs": 80,
#     }
# )



hyper_param_epoch = 5
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


# train_data_set=torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transforms_train)
train_data_set=torchvision.datasets.ImageFolder(root="./archive/train",transform=transforms_train)
train_loader=DataLoader(train_data_set,batch_size=hyper_param_batch,shuffle=True) #배치 단위로 loader

# test_data_set= torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transforms_test)
test_data_set=torchvision.datasets.ImageFolder(root="./archive/test",transform=transforms_train)
test_loader=DataLoader(test_data_set,batch_size=hyper_param_batch,shuffle=False)


if not (train_data_set.classes ==test_data_set.classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model, Loss, Optimizer
#binarize=True이면 XNOR_Net++ False면 일반 ResNet18
net = resnet18_preact_bin(num_classes=10, output_height=224, output_width=224, binarize=True).to(device) #output_height=32 output_width=32
# net=resnet18_XNOR(num_classes=10).to(device)
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
        
    # # log metrics to wandb
    # wandb.log({ "train loss": train_loss, "train acc":train_accuracy},step=epoch)

    
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
    
    # # log metrics to wandb
    # wandb.log({ "test loss": test_loss,"test acc":test_accuracy},step=epoch)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%,Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
print('Finished Training')

#===================================================================================================================
# wandb.finish()

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

#=================================================================================================================
# 모델을 평가 모드로 전환
net.eval()

#weight,activation(feature),scaling factor의 분포를 그래프로 분석
#몇가지 레이어의 weight를 추출
# 첫 번째 블록의 첫 번째 conv 레이어 (layer1[0].conv1)의 weights와 scaling factors 추출
block1_conv1_weights = net.layer3[0].conv2.weight.data.cpu().numpy().flatten()
block1_conv1_alpha = net.layer3[0].conv2.alpha.data.cpu().numpy().flatten()
block1_conv1_beta = net.layer3[0].conv2.beta.data.cpu().numpy().flatten()
block1_conv1_gamma = net.layer3[0].conv2.gamma.data.cpu().numpy().flatten()


# 히스토그램을 그려서 분포 시각화
plt.figure(figsize=(18, 6))

# Weights 분포 시각화
plt.subplot(1, 4, 1)
plt.hist(block1_conv1_weights, bins=100, alpha=0.75, color='blue')
plt.title('weights')
plt.xlabel('Weight value')
plt.ylabel('Frequency')

# Scaling factor (alpha) 분포 시각화
plt.subplot(1, 4, 2)
plt.hist(block1_conv1_alpha, bins=100, alpha=0.75, color='red')
plt.title('alpha')
plt.xlabel('Alpha value')
plt.ylabel('Frequency')

# Scaling factor (beta) 분포 시각화
plt.subplot(1, 4, 3)
plt.hist(block1_conv1_beta, bins=100, alpha=0.75, color='green')
plt.title('beta')
plt.xlabel('Beta value')
plt.ylabel('Frequency')

# Scaling factor (gamma) 분포 시각화
plt.subplot(1, 4, 4)
plt.hist(block1_conv1_gamma, bins=100, alpha=0.75, color='purple')
plt.title('gamma')
plt.xlabel('Gamma value')
plt.ylabel('Frequency')

plt.savefig('weight_distribution.jpg', format='jpg')

plt.tight_layout()
plt.show()

# `layer3`을 통과한 후의 활성화 값을 저장할 리스트
activations_layer3 = []

# 활성화 값을 추출하기 위해 hook을 정의
def hook_fn(module, input, output):
    activations_layer3.append(output.cpu().detach().numpy().flatten())

# `layer3`에 hook을 등록
hook = net.layer3.register_forward_hook(hook_fn)

# 데이터 로더에서 첫 배치를 통과시켜 활성화 값을 추출
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        _ = net(images)  # forward pass를 통해 hook에서 활성화 값을 추출
        break  # 한 배치만 사용

# hook 제거
hook.remove()

# 추출한 활성화 값을 numpy 배열로 변환
activations_layer3 = np.concatenate(activations_layer3)

# 히스토그램을 그려서 분포 시각화 및 저장
plt.figure(figsize=(6, 6))
plt.hist(activations_layer3, bins=50, alpha=0.75, color='blue')
plt.title('Distribution of activations in layer3')
plt.xlabel('Activation value')
plt.ylabel('Frequency')

# 그래프를 JPG 파일로 저장
plt.savefig('activations_layer3_histogram.jpg', format='jpg')

# 그래프를 화면에 표시
plt.show()

