import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import seaborn as sns

from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

test_dir = "/home/lordgrim/Work/Courses/CS6910/A1/5/test/"
img_h,img_w = 84,84

## The best performing model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv1a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv2a = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)


        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=33)      # change out_features according to number of classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv1a(x)))

        x = F.relu(self.conv2(x))
        x = self.pool(self.conv2a(x))

        x = F.relu(self.conv3(x))
        x = self.pool(self.conv3a(x))

        x = F.avg_pool2d(x, kernel_size=x.shape[2:])

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## Load the model weights and check architecture        
path_model = "/home/lordgrim/Work/Courses/CS6910/A1/models/62_nice.pth"
net = Net()
print(net)

# transfer the model to GPU
if torch.cuda.is_available():
    net = net.cuda()
    
net.load_state_dict(torch.load(path_model))
summary(net,(3,img_h,img_w))

## Transforms and the loader for the model
loader = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
def image_loader(image):
    """load image, returns cuda tensor"""
    image = Image.open(image)
    np_img = np.array(image)
    image = loader(image).float()
    return image.cuda(), np_img  #assumes that you're using GPU

## For layer 1
max_act = {0:[],1:[],2:[]}
img_patch_all = {0:[],1:[],2:[]}

for class_label in os.listdir(test_dir):
    for img_path in os.listdir(test_dir + class_label):
        ## Using activation hooks seems like a more elegant way than to modify the model  
        #  to return as outputs the intermediate activations
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        net.conv1.register_forward_hook(get_activation('conv1'))
        inp, img = image_loader(test_dir+class_label+"/"+img_path)

        out = net(torch.reshape(inp, (-1, 3, img_h, img_w) ))
        act = activation['conv1'].squeeze().cpu()

        # Receptive field
        # We are using square kernel only, so x=y
        rc_field = net.conv1.kernel_size[0] # 3 here for the 1st layer

        act_10= act[9,:,:] # 10nd filter
        act_17 = act[16,:,:] # 17th filter
        act_31 = act[30,:,:] #31st filter

        ## For each filter in layer 1
        for j,act_val in enumerate([act_10,act_17,act_31]):

            ## To get the top5 max in the array, we need to flatten and then find
            top5 = torch.topk(torch.flatten(act_val),5)
            max_vals = top5[0]
            ans = top5[1]

            ## Get x,y indices of the top 5
            max_x = ans%act_val.shape[0]
            max_y = ans//act_val.shape[1]

                       
            for i,(x,y) in enumerate(zip(max_x,max_y)):                
                x, y = int(x.numpy()), int(y.numpy())

                x1 = x 
                x2 = x + rc_field
                y1 = y 
                y2 = y + rc_field
                img_patch_d = [test_dir+class_label+"/"+img_path,(x1,x2,y1,y2)]

                ## Store the top 5 values and the corresponding image (patch, and file path)
                max_act[j].append(max_vals[i])
                img_patch_all[j].append(img_patch_d)

## Iterate over each layer
for j,layer in enumerate(['layer_10','layer_17','layer_31']):
    plt.figure(figsize=(10,10))
    activations = np.array(max_act[j])
    img_patches = img_patch_all[j]

    ## Find the indices of the global absolute top 5 values
    temp = np.argpartition(-activations, 5)
    result_args = temp[:5]
    print(result_args)

    f, axarr = plt.subplots(2,5, figsize=(15,15))
    ## For each of the top 5 plot the figure with the image patch in the report
    for i,index in enumerate(result_args):
        vals = img_patches[index]
        img = np.array(Image.open(vals[0]))
        x1, x2, y1, y2 = vals[1]
        img_patch = img[y1:y2,x1:x2]

        axarr[0,i].axis('off',aspect="auto")
        axarr[0,i].imshow(img)
        axarr[1,i].axis('off',aspect="auto")
        axarr[1,i].imshow(img_patch)

        plt.axis('off')
    plt.show()
    f.savefig('conv1_{}.png'.format(layer))
    

