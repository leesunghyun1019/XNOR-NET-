#net=resnet18_XNOR(num_classes=10, output_height=224, output_width=224, binarize=True).to(device)
#ResNet18 기반으로 XNOR-Net++ 구현
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinActiv(torch.autograd.Function):
  @staticmethod
  def forward(ctx,input):
    ctx.save_for_backward(input)
    output=input.sign()
    return output
  
  @classmethod
  def Mean(cls,input):
    return torch.mean(input.abs(),1,keepdim=True)

  @staticmethod
  def backward(ctx,grad_output):
    input, =ctx.saved_tensors
    grad_input=grad_output.clone()
    grad_input[input.ge(1)]=0
    grad_input[input.le(-1)]=0
    return grad_input

binactive=BinActiv.apply

class BinConv2d(nn.Module):
  def __init__(self,input_channels,output_channels,kernel_size=-1,stride=1,padding=0,
               groups=1,bias=False,dilation=1):
    super(BinConv2d,self).__init__()
    self.stride=stride
    self.padding=padding
    self.shape=(output_channels,input_channels,kernel_size,kernel_size)
    self.kernel_size=kernel_size
    self.weight=nn.Parameter(torch.rand(self.shape)*0.001,requires_grad=True)
  
  def forward(self,x):
    #Binarize input
    x=binactive(x) #x의 size가 (64,64,56,56)
    
    #scaling factor K
    A = BinActiv.Mean(x)  
    k=torch.ones(1,1,self.kernel_size,self.kernel_size).mul(1/(self.kernel_size**2)).to(x.device)  # Ensure k is on the same device as x
    K = F.conv2d(A, k, stride=self.stride, padding=self.padding)#K의 size은 (64,1,28,28)
    
    #scaling factor alpha
    n=self.weight[0].nelement()
    alpha=self.weight.norm(1,3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n).expand_as(self.weight)#alpha의 크기는 (128,64,3,3)

    #Binarize Weight
    real_weight=self.weight #weights의 크기는 (128,64,3,3)
    #mean_weights와 centered_weights는 (batchnorm)한 느낌이다
    mean_weights = real_weight.mul(-1).mean(dim=1, keepdim=True).expand_as(self.weight).contiguous()
    centered_weights = real_weight.add(mean_weights) 
    cliped_weights = torch.clamp(centered_weights, -1.0, 1.0) #cliped_weights는 tanh 함수를 의미 
    signed_weights = torch.sign(centered_weights).detach() - cliped_weights.detach() + cliped_weights
    binary_weights = signed_weights

    binary_weights=binary_weights.mul(alpha)
    x = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding) #x의 크기는 (64,128,28,28)
    return x.mul(K)


def conv3x3(in_planes,out_planes,stride=1,groups=1,dilation=1,binarize=False): #kernel_size is 3x3
    
    if binarize:
        return BinConv2d(in_planes,out_planes,kernel_size=3,stride=stride,
                     padding=dilation,groups=groups,bias=False,dilation=dilation)
    else:
        return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=dilation,groups=groups,
                     bias=False,dilation=dilation)
  
def conv1x1(in_planes,out_planes,stride=1,binarize=False):#kernel_size is 1x1
    if binarize:
        return BinConv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)
    else:
        return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

class BasicBlock(nn.Module):
  expansion=1

  def __init__(self,inplanes,planes,stride=1,downsample=None,groups=1,base_width=64,dilation=1,norm_layer=None,binarize=False):
    super(BasicBlock,self).__init__()
    if norm_layer is None:
      norm_layer=nn.BatchNorm2d
    if groups !=1 or base_width !=64:
      raise ValueError('BasicBlock only suppoets groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

    #Both self.conv1 and self.downsample layers downsample the input when stride !=1
    # self.bn1=norm_layer(inplanes)
    self.bn1=norm_layer(inplanes)
    self.relu=nn.ReLU(inplace=True)
    self.conv1=conv3x3(inplanes,planes,stride,binarize=binarize)
    self.bn2=norm_layer(planes)
    self.bn3=norm_layer(planes)
    self.conv2 = conv3x3(planes, planes,binarize = binarize)
    self.downsample = downsample
    self.stride = stride

  def forward(self,x):
    identity=x
    out=self.bn1(x)
    if self.downsample is not None:
      identity=self.downsample(out) #downsample 여부
    out=self.conv1(out)
    out = self.relu(out)
    out = self.bn2(out)
    out = self.conv2(out)
    out = self.bn3(out)
    out += identity
    out = self.relu(out)
    return out
  

class ResNet(nn.Module):
  def __init__(self,block,layers,num_classes=1000,zero_init_residual=False,
               groups=1,width_per_group=64,replace_stride_with_dilation=None,
               norm_layer=None,output_height=224,output_width=224,binarize=False):
    super(ResNet,self).__init__()
    if norm_layer is None:
      norm_layer=nn.BatchNorm2d
    self._norm_layer=norm_layer

    self.inplanes=64 #집중
    self.dilation=1

    if replace_stride_with_dilation is None:
       # each element in the tuple indicates if we should replace
       # the 2x2 stride with a dilated convolution instead
       replace_stride_with_dilation=[False,False,False]
    if len(replace_stride_with_dilation) !=3:
      raise ValueError("replace_stride_with_dilation should be None"
                      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    self.groups=groups
    self.base_width=width_per_group
    self.conv1=nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False) #여기서는 3이 입력 채널, self.inplanes 출력채널
    self.bn1=norm_layer(self.inplanes)
    self.relu=nn.ReLU(inplace=True)
    self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    self.layer1 = self._make_layer(block, 64, layers[0], binarize = binarize) #planes는 출력채널의 수를 정한다

    self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], binarize = binarize)

    self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], binarize = binarize)

    self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], binarize = binarize)
    self.bn5=norm_layer(512*block.expansion)
    self.avgpool=nn.AdaptiveAvgPool2d((1,1))
    self.fc=nn.Linear(512*block.expansion,num_classes)

    for m in self.modules():#가중치 초기화
      if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        # if isinstance(m, Bottleneck):
        #   nn.init.constant_(m.bn3.weight, 0)
        if isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self,block,planes,blocks,stride=1,dilate=False,binarize=True):
    norm_layer=self._norm_layer
    downsample=None
    previous_dilation=self.dilation

    if dilate: # 팽창률을 통해 필터의 간격을 조정하며, 필터가 더 큰 영역을 커버, 해상도 유지
      self.dilation *=stride
      stride=1

    if stride !=1 or self.inplanes != planes* block.expansion: #planes* block.expansion는 해당 블록의 최종 출력 채널의 수를 결정한다
      #downsample이란 무엇인가?  이전 출력의 채널 수가 현재 입력 채널의 수와 같지 않다면!!
      # downsample=nn.Sequential(
      #     nn.AvgPool2d(kernel_size=2,stride=stride),
      #     conv1x1(self.inplanes,planes*block.expansion,1,binarize=False),
      #             norm_layer(planes*block.expansion),
      # )
      downsample=nn.Sequential(
          conv1x1(self.inplanes,planes*block.expansion,stride,binarize=False),
                  norm_layer(planes*block.expansion),
      )

    layers=[]
    #첫번째 블록에서만 다운샘플링이 필요할 수 있다.
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                        previous_dilation, norm_layer, binarize=binarize))

    self.inplanes = planes * block.expansion #최종 출력 채널의 수가 다음 블록의 입력 채널 수가 된다
    #first layers에서는 planes 64 였고 block.expansion은 1이므로 self.inplanes=64

    for _ in range(1, blocks): #두번째 블록부터는 다운샘플링이 필요없다
        layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                            dilation=self.dilation, norm_layer=norm_layer, binarize=binarize))


    return nn.Sequential(*layers)

  def _forward_impl(self, x):

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x=self.fc(x)

    return x

  def forward(self, x):
    return self._forward_impl(x)


def resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18_XNOR(progress=True, **kwargs):
    return resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=progress, **kwargs)