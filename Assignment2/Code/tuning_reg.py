import torch, os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchsummary import summary
import numpy as np
import time
import matplotlib.pyplot as plt




########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].

# Apply necessary image transfromations here 

transform = transforms.Compose([ 
                                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
print(transform)

data_root = "/home/ubuntu/t1/5"
train_data_dir = data_root + '/train' # put path of training dataset
val_data_dir = data_root + '/val' # put path of validation dataset
test_data_dir = data_root + '/test' # put path of test dataset

trainset = torchvision.datasets.ImageFolder(root= train_data_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

valset = torchvision.datasets.ImageFolder(root= val_data_dir, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                         shuffle=False, num_workers=2)

testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

########################################################################
# Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# <<<<<<<<<<<<<<<<<<<<< EDIT THE MODEL DEFINITION >>>>>>>>>>>>>>>>>>>>>>>>>>
# Try experimenting by changing the following:
# 1. number of feature maps in conv layer
# 2. Number of conv layers
# 3. Kernel size
# etc etc.,

num_epochs = 80         # desired number of training epochs.
learning_rate = 0.001


def experiment(config):
    
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


    ################### DO NOT EDIT THE BELOW CODE!!! #######################

    net = Net()
    print(net)
    summary(net.cuda(), (3, 84, 84))

    # transfer the model to GPU
    if torch.cuda.is_available():
        net = net.cuda()

    ########################################################################
    # Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    weight_decay = config['val'] ## 5e-4
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    num_params = np.sum([p.nelement() for p in net.parameters()])
    print(num_params, ' parameters')

    ########################################################################
    # Train the network
    # ^^^^^^^^^^^^^^^^^^^^

    def train(epoch, trainloader, optimizer, criterion,valloader):
        running_loss = 0.0
        for i, data in enumerate((trainloader), 0): #tqdm
            # get the inputs
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        running_loss1 = 0.0
        for i, data in enumerate((valloader), 0): #tqdm
            # get the inputs
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            running_loss1 += loss.item()

        return ( running_loss/len(trainloader) , running_loss1/len(valloader) )

    ########################################################################
    # Let us look at how the network performs on the test dataset.

    def test(testloader, model, file=False):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()        
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if file == False:
            print('Accuracy of the network on the test images: %d %%' % (
                                        100 * correct / total))
        else:
            print('Accuracy of the network on the test images: %d %%' % (
                                        100 * correct / total), file=file)
            
        return correct / total

    #########################################################################
    # get details of classes and class to index mapping in a directory
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


    def classwise_test(testloader, model, file=False):
    ########################################################################
    # class-wise accuracy

        classes, _ = find_classes(train_data_dir)
        n_class = len(classes) # number of classes

        class_correct = list(0. for i in range(n_class))
        class_total = list(0. for i in range(n_class))
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()        
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(n_class):
            if file == False:
                print('Accuracy of %10s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
            else:
                print('Accuracy of %10s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]),file=file)


    print('Start Training')
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./models/{}'.format(config['name']), exist_ok=True)
    
    train_acc_all = []
    val_acc_all = []

    train_loss_all = []
    val_loss_all = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('epoch ', epoch + 1)
        train_loss, val_loss = train(epoch, trainloader, optimizer, criterion, valloader)

        accuracy = test(valloader, net)
        train_accuracy = test(trainloader, net)

        train_acc_all.append(train_accuracy)
        val_acc_all.append(accuracy)

        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)
                        

    print('Finished Training')
    return train_acc_all,val_acc_all,train_loss_all,val_loss_all

values = [1e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6]

for i,val in enumerate(values):
    config = {}
    config['name'] = 'reg_exp'
    config['val'] = val

    train_acc_all,val_acc_all,train_loss_all,val_loss_all = experiment(config)

    np.savez('acc_mat_{}.npz'.format(i), train=np.array(train_acc_all), test=np.array(val_acc_all))
    np.savez('loss_mat_{}.npz'.format(i), train=np.array(train_loss_all), test=np.array(val_loss_all))


## Some helper code to plot the results in the Report

## Plot train vs val accuracy curves
folder_path = "./reg_results/"
values = [1e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6]
for i,val in enumerate(values):
    result = np.load(folder_path+"acc_mat_"+str(i)+".npz")

    train_acc_all = result['train']
    val_acc_all = result['test']
    plt.figure()
    plt.plot(np.arange(len(train_acc_all)),train_acc_all)
    plt.plot(np.arange(len(train_acc_all)),val_acc_all)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.yticks(np.arange(0,1.1,0.1))
    plt.savefig('train_times_{}.png'.format(i))
    # plt.show()
    print(str(val)+" & "+str(round(np.max(train_acc_all)*100,2))+" & "+str(round(np.max(val_acc_all)*100,2)) )

## Plot train vs val loss curves
folder_path = "./reg_results/"
values = [1e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6]
for i,val in enumerate(values):
    result = np.load(folder_path+"loss_mat_"+str(i)+".npz")

    train_acc_all = result['train']
    val_acc_all = result['test']
    plt.figure()
    plt.plot(np.arange(len(train_acc_all)),train_acc_all)
    plt.plot(np.arange(len(train_acc_all)),val_acc_all)
    plt.title('Losses during training')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('train_times_acc_{}.png'.format(i))
    # plt.show()
