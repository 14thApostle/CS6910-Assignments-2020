import torch, os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchsummary import summary
import numpy as np
import time

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].

# Apply necessary image transfromations here 

transform = transforms.Compose([ torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
print(transform)


data_root = "../5"
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

num_epochs = 100        # desired number of training epochs.
learning_rate = 0.001   


########################################################################
# Train the network
# ^^^^^^^^^^^^^^^^^^^^

def train(epoch, trainloader, optimizer, criterion):
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

    print('epoch %d training loss: %.3f' %
            (epoch + 1, running_loss / (len(trainloader))))
    
########################################################################
# Let us look at how the network performs on the test dataset.

def test(testloader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in (testloader): #tqdm
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

#     print('Accuracy of the network on the test images: %d %%' % (
#                                     100 * correct / total))
    return 100 * correct / total

#########################################################################
# get details of classes and class to index mapping in a directory
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def classwise_test(testloader, model, f):
########################################################################
# class-wise accuracy

    classes, _ = find_classes(train_data_dir)
    n_class = len(classes) # number of classes

    class_correct = list(0. for i in range(n_class))
    class_total = list(0. for i in range(n_class))
    with torch.no_grad():
        for data in (testloader): #tqdm
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(n_class):
        if f == False:
            print('Accuracy of %10s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        else:
            print('Accuracy of %10s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]),file=f)



def experiment(net,config):

    ################### DO NOT EDIT THE BELOW CODE!!! #######################

    #net = ResNet()
    # net = Net()
    # net = model.load()
    # transfer the model to GPU
    if torch.cuda.is_available():
        net = net.cuda()

    ########################################################################
    # Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    num_params = np.sum([p.nelement() for p in net.parameters()])
    print(num_params, ' parameters')


    print('Start Training')
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./models/{}'.format(config['name']), exist_ok=True)
    best_accuracy = 0

    t1 = time.time()
    val_acc = []
    train_acc = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        train(epoch, trainloader, optimizer, criterion)
        train_accuracy = test(trainloader, net)
        accuracy = test(valloader, net)
        train_acc.append(train_accuracy)
        val_acc.append(val_acc)
        print("Train - ",round(train_accuracy),"%")
        print("Val - ",round(accuracy),"%","\n")
    #     classwise_test(valloader, net)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(net.state_dict(), './models/{}/{}'.format(config['name'],config['val'])+'.pth') 

            f = open('./models/{}/{}.txt'.format(config['name'],config['val'],epoch), "w+")
            print("Train - ",round(train_accuracy),"%",file=f)
            print("Val - ",round(accuracy),"%",file=f)
            test_acc = test(testloader, net)
            print("Val - ",round(test_acc),"%","\n",file=f)
            classwise_test(testloader, net,f)



            
    # os.rename('./models/{}'.format(t1)+'.pth','./models/{}_{}'.format(best_accuracy,t1) +'.pth')
    print('performing test')
    test_acc = test(testloader, net)
    print(test_acc)
    classwise_test(testloader, net, False)

    print('Finished Training')

class Net1(nn.Module):
    def __init__(self, val):
        super(Net1, self).__init__()

        self.val = val

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=33)
        self.fc3 = nn.Linear(in_features=33, out_features=33)      # change out_features according to number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
            
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net2(nn.Module):
    def __init__(self, val):
        super(Net2, self).__init__()

        self.val = val

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=33)      # change out_features according to number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
            
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net3(nn.Module):
    def __init__(self, val):
        super(Net3, self).__init__()

        self.val = val

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5)

        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=33)      # change out_features according to number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
            
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    config = {}
    config['name'] = "n_filter"
    for i,k in enumerate([Net1,Net2,Net3]):
        print("Running ",i)
        config['val'] = i
        net = k(0)
        summary(net.cuda(),(3,84,84))
        experiment(net,config)
