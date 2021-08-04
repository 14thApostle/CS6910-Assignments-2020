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

test_dir = "/home/lordgrim/Work/Courses/CS6910/A1/Q2_testset/"
img_h,img_w = 84,84

## The best performing net
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
        self.fc3 = nn.Linear(in_features=256, out_features=33)

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
        
# transfer the model to GPU
if torch.cuda.is_available():
    net = net.cuda()

## Load the model, and verify architecture
path_model = "/home/lordgrim/Work/Courses/CS6910/A1/models/62_nice.pth"
net = Net()
print(net)
net.load_state_dict(torch.load(path_model))
summary(net,(3,img_h,img_w))

loader = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5,0.5,0.5], 
                                                std=[0.5, 0.5, 0.5])])

def image_loader(image):
    """load image, returns cuda tensor"""
    np_img = np.array(image)
    image = loader(image).float()
    return image.cuda(), np_img  #assumes that you're using GPU

N = 11 ## should be odd
## Folder to save imgs and corresponding heatmaps before plotting
os.makedirs("./occlusion_exp/{}".format(N), exist_ok = True) 

## Iterate over the test set
for img_path in os.listdir(test_dir):
    x,img = image_loader(Image.open(test_dir+img_path))
    conf_arr = np.zeros((img_h,img_w))

    n = (N-1)//2
    for i in range(img_w):
        for j in range(img_h):
            # Create a copy of the input img and calulate window size to grey out
            new_img = img.copy()
            x1 = max(0,i-n)
            x2 = min(i+n+1,img_w)
            y1 = max(0,j-n)
            y2 = min(j+n+1,img_h)
            # Grey out the part of input img
            new_img[y1:y2,x1:x2,:] = 211

            ## Run the model through the new occluded image 
            x1 = image_loader(Image.fromarray(new_img))[0]
            out1 = net(torch.reshape(x1, (-1, 3, img_h, img_w) ))
            y_pred1 = np.argmax(out1.cpu().detach().numpy())
            ## Softmax the new model output and append the probability 
            output = nn.functional.softmax(out1, dim=1)
            prob = output.tolist()[0][y_pred1]
            conf_arr[j,i] = prob

    imgplot = sns.heatmap(conf_arr, xticklabels=False, yticklabels=False)
    figure = imgplot.get_figure()
    figure.savefig("./occlusion_exp/{}/heatmap_{}.png".format(N,img_path[:-4] ) ) 
    Image.fromarray(img).save("./occlusion_exp/{}/{}.png".format(N,img_path[:-4] )) 

folder_path = "./occlusion_exp/{}/".format(N)
print("No of imgs and heatmaps",len(os.listdir(folder_path)))


## Code to generate the multiple plots attached in the report
i = 0
f, axarr = plt.subplots(28,2, figsize=(50,150))
for file_path in os.listdir(folder_path):
    if file_path.startswith('heatmap'):      
        heatmap = Image.open(folder_path+file_path)
        img = Image.open(folder_path+file_path.split('heatmap_')[-1])
        axarr[i,0].axis('off',aspect="auto")
        axarr[i,0].imshow(img)
        axarr[i,1].axis('off',aspect="auto")
        axarr[i,1].imshow(heatmap)
        i+=1
f.savefig('final_random_11.jpg')