import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torchvision.utils import make_grid, save_image

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


net = ResNet50()
net = net.to(device)

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']


'''cifar data'''
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=20, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

loader = iter(testloader)
images, labels = loader.next()
# 여기서 [] 에다가어떤 이미지를 넣을지 생각하면 댐. 
Test_image = images[8].unsqueeze(0)
Test_image = Test_image.cuda()
image_test = Test_image.cpu().numpy() # 1,3,224,224 to 3,224,224

#plt.imshow(np.transpose(image_test[0],(1,2,0))) # 이미지 보여주기 224,224,3 늘려준 차원을 다시 인덱싱으로 없애줌. 
#plt.show()

image_test = np.transpose(image_test[0],(1,2,0)) # grad cam 보여주기 위해서 트랜즈 포즈 시킴.



'''
이미지 보여주는 것 
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
imshow(torchvision.utils.make_grid(images))
'''


# layer 변경
finalconv_name = 'layer3'
feature_blobs = []
backward_feature = []

def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data)
    
# Grad-CAM
def backward_hook(module, input, output):
    backward_feature.append(output[0])
    
net._modules.get(finalconv_name).register_forward_hook(hook_feature)
net._modules.get(finalconv_name).register_backward_hook(backward_hook)

net.cuda()

params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].cpu().detach().numpy()) 

logit = net(Test_image)


score = logit[:, 3].squeeze() #클래스 번호 입력하면 됌
score.backward(retain_graph = True) 
activations = feature_blobs[0].to(device) 
gradients = backward_feature[0] 
b, k, u, v = gradients.size()
alpha = gradients.view(b, k, -1).mean(2) 
weights = alpha.view(b, k, 1, 1)

from torch.autograd import Function


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        

        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        
        # forward에서 저장된 saved tensor를 불러오기
        input_img, output = self.saved_tensors
        grad_input = None
        positive_mask_1 = (input_img > 0).type_as(grad_output)
        
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input
  
class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))
        return output

def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

grad_cam_map = (weights*activations).sum(1, keepdim = True) 
grad_cam_map = F.relu(grad_cam_map) 
grad_cam_map = F.interpolate(grad_cam_map, size=(224, 224), mode='bilinear', align_corners=False) 

map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data


grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) 


#layer 저장
cv2.imwrite(os.path.join( "Grad_CAM_heatmap_layer3.jpg"), grad_heatmap)

grad_heatmap = np.float32(grad_heatmap) / 255



grad_result = grad_heatmap
grad_result = grad_result / np.max(grad_result)
grad_result = np.uint8(255 * grad_result)

cv2.imwrite(os.path.join( "Grad_Result_layer3.jpg"), grad_result)