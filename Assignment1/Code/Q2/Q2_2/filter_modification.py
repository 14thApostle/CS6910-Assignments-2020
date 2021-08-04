import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import seaborn as sns

import copy

from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

test_dir = "/home/lordgrim/Work/Courses/CS6910/A1/5/test/"
img_h,img_w = 84,84

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
summary(net.cuda(),(3,img_h,img_w))

## Transforms and the loader for the model
loader = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
def image_loader(image):
    """load image, returns cuda tensor"""
    image = Image.open(image)
    np_img = np.array(image)
    image = loader(image).float()
    return image.cuda(), np_img  #assumes that you're using GPU

## Classes list indexed for checking accuracy
classes_list = {}
rev_classes_list = {}
for i,class_name in enumerate( sorted(os.listdir(test_dir)) ):
    classes_list[class_name] = i
    rev_classes_list[i] = class_name
print(classes_list)

## For layer 1
os.makedirs("Misses/layer1/",exist_ok=True)
for filter_n in [9,30]:
    save_folder = "Misses/layer1/{}/".format(filter_n)
    os.makedirs(save_folder,exist_ok=True)

    ## Make a copy of the net, of which we will be turning filters off.
    net1 = Net()
    net1.load_state_dict(copy.deepcopy(net.state_dict()))
    net1.cuda()
    net1.conv1.weight[filter_n,:,:,:] = 0

    misses = 0
    correct_pred = 0
    total_imgs = 3300

    for class_label in os.listdir(test_dir):
        gt_class_id = classes_list[class_label]
        for img_path in os.listdir(test_dir + class_label):

            inp, img = image_loader(test_dir+class_label+"/"+img_path)

            out = net(torch.reshape(inp, (-1, 3, img_h, img_w) ))
            prob = torch.nn.functional.softmax(out)
            class_id = torch.argmax(prob)
            prob_val = prob[0][class_id]
            ## Only proceed if img was correctly classified by the original model
            if gt_class_id == class_id:
                correct_pred += 1

                ## Find the prediction of the new model with filter turned off
                out1 = net1(torch.reshape(inp, (-1, 3, img_h, img_w) ))
                prob1 = torch.nn.functional.softmax(out1)
                class_id1 = torch.argmax(prob1)
                prob_val1 = prob1[0][class_id1]

                ## Proceed if the new model misclassified the input 
                if class_id != class_id1:
 
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    misses += 1

                    # Plot the image and its ground truth and prediction
                    plt.axis("off")
                    plt.imshow(img)
                    plt.xlabel("img",color="green")
                    ax.text(0.1, -0.07, 'Prediction : {}'.format(rev_classes_list[int(class_id1.cpu())]),
                            verticalalignment='bottom',
                            transform=ax.transAxes,
                            color='red', fontsize=10)
                    ax.text(0.1, -0.12, 'Ground truth : {}'.format(class_label),
                        verticalalignment='bottom',
                        transform=ax.transAxes,
                        color='green', fontsize=10)
                    plt.savefig(save_folder+str(misses)+".png")

    print("# Filter number {} # set to zero".format(filter_n))
    print("Missed : ",misses)
    print("Accuracy dropped from {} to {}".format(correct_pred/total_imgs, (correct_pred-misses)/total_imgs))
    print("\n")

def plot_multiple_misses(path_dir):
    '''
        Some helper code to plot the figures for the report
    '''
    # path_dir = "/home/lordgrim/Work/Courses/CS6910/A1/Misses_for_report/"
    for layer in os.listdir(path_dir):
        for filter_number in os.listdir(path_dir+layer):
            size = len(os.listdir( (path_dir+layer+"/"+filter_number)))
            f, axarr = plt.subplots(1,size, figsize=(70,35))
            for i,img_path in enumerate(os.listdir(path_dir+layer+"/"+filter_number)):
                img = Image.open(path_dir+layer+"/"+filter_number+"/"+img_path)
                axarr[i].axis('off',aspect="auto")
                f.subplots_adjust(wspace=None, hspace=None)
                axarr[i].imshow(img)
            plt.show()
            f.savefig('{}img_{}_{}.png'.format(path_dir,layer,filter_number))