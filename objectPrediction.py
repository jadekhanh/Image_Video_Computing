# -*- coding: utf-8 -*-
"""CS585_PS2_Final.ipynb
#**Q1: Cifar-10 class predictions**

# Mount Google Drive

This will allow the Colab machine to access Google Drive folders by mounting the drive on the machine. You may be asked to copy and paste an authentication code.
"""

from google.colab import drive
drive.mount('/content/gdrive/')

!ls

"""# Change directory to allow imports


As noted above, you should create a Google Drive folder to hold all your assignment files. You will need to add this code to the top of any python notebook you run to be able to import python files from your drive assignment folder (you should change the file path below to be your own assignment folder).
"""

import os
if not os.path.exists("/content/gdrive/My Drive/Colab Notebooks/CS_585_PS2"):
    os.makedirs("/content/gdrive/My Drive/Colab Notebooks/CS_585_PS2")
os.chdir("/content/gdrive/My Drive/Colab Notebooks/CS_585_PS2")

!ls # Check if this is your PS2 folder

"""# Set up GPU and PyTorch

First, ensure that your notebook on Colaboratory is set up to use GPU. After opening the notebook on Colaboratory, go to Edit>Notebook settings, select Python 3 under "Runtime type," select GPU under "Hardware accelerator," and save.

Next, install PyTorch:
"""

!pip3 install torch torchvision

"""Make sure that pytorch is installed and works with GPU:"""

import torch
a = torch.Tensor([1]).cuda()
print(a)

torch.cuda.is_available()

# Commented out IPython magic to ensure Python compatibility.
# imports and useful functions

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import copy
import csv
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os.path
from tqdm.notebook import tqdm
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.manual_seed(111)


class CIFAR10Test(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None,
    ):
        super(CIFAR10Test, self).__init__(root, transform=transform)

        image_filename = os.path.join(root, 'cifar10_test_images.npy')
        images = np.load(image_filename)

        assert len(images.shape) == 4
        assert images.shape[0] == 2000
        assert images.shape[1] == 32
        assert images.shape[2] == 32
        assert images.shape[3] == 3

        self.data = images

    def __getitem__(self, index):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)


def calculate_accuracy(dataloader, model, is_gpu):
    """ Util function to calculate val set accuracy,
    both overall and per class accuracy
    Args:
        dataloader (torch.utils.data.DataLoader): val set
        is_gpu (bool): whether to run on GPU
    Returns:
        tuple: (overall accuracy, class level accuracy)
    """
    correct = 0.
    total = 0.
    predictions = []

    class_correct = list(0. for i in range(TOTAL_CLASSES))
    class_total = list(0. for i in range(TOTAL_CLASSES))

    # Check out why .eval() is important!
    # https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744/2
    model.eval()

    with torch.no_grad():
      for data in dataloader:
          images, labels = data
          if is_gpu:
              images = images.cuda()
              labels = labels.cuda()
          outputs = model(Variable(images))
          _, predicted = torch.max(outputs.data, 1)
          predictions.extend(list(predicted.cpu().numpy()))
          total += labels.size(0)
          correct += (predicted == labels).sum()

          c = (predicted == labels).squeeze()
          for i in range(len(labels)):
              label = labels[i]
              class_correct[label] += c[i].cpu()
              class_total[label] += 1

    class_accuracy = 100 * np.divide(class_correct, class_total)
    return 100*correct/total, class_accuracy


def run_secret_test(dataloader, model, is_gpu):
    predictions = []
    model.eval()

    with torch.no_grad():
      for images in dataloader:
          if is_gpu:
              images = images.cuda()
          outputs = model(Variable(images))
          predicted = torch.softmax(outputs, dim=1).cpu().numpy()
          predictions.extend(list(predicted))

    return predictions

"""
Training an image classifier
----------------------------

We will do the following steps in order:

1. Load the randomized CIFAR10 training, validation and test datasets using
   torchvision. Use torchvision.transforms to apply transforms on the
   dataset.
2. Define a Convolution Neural Network - BaseNet
3. Define a loss function and optimizer
4. Train the network on training data and check performance on val set.
   Plot train loss and validation accuracies.
5. Try the network on test data and create .npy file for submission to Gradescope"""

import os
print(os.getcwd())

# <<TODO>>: Based on the val set performance, decide how many
# epochs are appropriate for your model.
# ---------
EPOCHS = 15
# ---------

IS_GPU = True
TEST_BS = 256
TOTAL_CLASSES = 10
TRAIN_BS = 32
PATH_TO_CIFAR10 = "cifar10_splits/"
PATH_TO_CIFAR10_TEST = "cifar10_splits/"

"""1.**Loading CIFAR-10**

We will load the CIFAR-10 dataset with builtin dataset loader from Torchvision. We also created our own train, validation and test splits. You can download them using this link: https://drive.google.com/file/d/1VkiwqowKrNMe6l3wV7-d6FIU3PENpcsI/view?usp=sharing . Upload the file to colab.

We provide a screenshot below on how and wwhere to upload the required data for both questions.
"""

plt.axis('off')
plt.imshow(Image.open('./upload_steps.png'))

!unzip -qqo /content/cifar10_splits.zip

# The output of torchvision datasets are PILImage images of range [0, 1].
# Using transforms.ToTensor(), transform them to Tensors of normalized range
# [-1, 1].


# <<TODO#1>> Use transforms.Normalize() with the right parameters to
# make the data well conditioned (zero mean, std dev=1) for improved training.
# <<TODO#2>> Try using transforms.RandomCrop() and/or transforms.RandomHorizontalFlip()
# to augment training data.
# After your edits, make sure that test_transform should have the same data
# normalization parameters as train_transform
# You shouldn't have any data augmentation in test_transform (val or test data is never augmented).
# ---------------------

dataset = torchvision.datasets.CIFAR10(root=PATH_TO_CIFAR10, train=True, download=True, transform=transforms.ToTensor())
mean = dataset.data.mean(axis=(0, 1, 2))/255
std = dataset.data.std(axis=(0, 1, 2))/255

train_transform = transforms.Compose(
    [
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])
# ---------------------

#DO NOT CHANGE any line below
train_dataset = torchvision.datasets.CIFAR10(root=PATH_TO_CIFAR10, train=True, download=True, transform=train_transform)
val_dataset = torchvision.datasets.CIFAR10(root=PATH_TO_CIFAR10, train=False, download=False, transform=test_transform)
test_dataset = CIFAR10Test(root=PATH_TO_CIFAR10_TEST, transform=test_transform)

val_dataset.data = np.load("cifar10_splits/cifar10_val_images.npy")
val_dataset.targets = np.load("cifar10_splits/cifar10_val_labels.npy")
test_dataset.data = np.load("cifar10_splits/cifar10_test_images.npy")

print("train_dataset data shape: ", np.array(train_dataset.data).shape)
print("train_dataset labels shape: ", np.array(train_dataset.targets).shape)
print()
print("val_dataset data shape: ", np.array(val_dataset.data).shape)
print("val_dataset labels shape:", np.array(val_dataset.targets).shape)

# check for Dataloader function: https://pytorch.org/docs/stable/data.html
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BS, shuffle=True, num_workers=2, drop_last=True)  #DO NOT CHANGE
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=TEST_BS, shuffle=False, num_workers=2, drop_last=False) #DO NOT CHANGE
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BS, shuffle=False, num_workers=2, drop_last=False) #DO NOT CHANGE

# The 10 classes for FashionMNIST
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

"""2.**Visualize CIFAR-10**

We will visualize some random images from the CIFAR-10 dataset.
"""

# Let us show some of the training images, for fun.
# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images[:16]))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(16)))

"""3.**Define a Convolution Neural Network**

Implement the BaseNet exactly. BaseNet consists of two convolutional modules (conv-relu-maxpool) and two linear layers. The precise architecture is defined below:

| Layer No.   | Layer Type  | Kernel Size | Input Dim   | Output Dim  | Input Channels | Output Channels |
    | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
    | 1 | conv2d | 5 | 32 | 28 | 3 | 6 |
    | 2 | relu | - | 28 | 28 | 6 | 6 |
    | 3 | maxpool2d | 2 | 28 | 14 | 6 | 6 |
    | 4 | conv2d | 5 | 14 | 10 | 6 | 16 |
    | 5 | relu | - | 10 | 10 | 16 | 16 |
    | 6 | maxpool2d | 2 | 10 | 5 | 16 | 16 |
    | 7 | linear | - | 1 | 1 | 400 | 200 |
    | 8 | relu | - | 1 | 1 | 200 | 200 |
    | 9 | linear | - | 1 | 1 | 200 | 10 |
"""

########################################################################
# We provide a basic network that you should understand, run and
# eventually improve
# <<TODO>> Add more conv layers
# <<TODO>> Add more fully connected (fc) layers
# <<TODO>> Add regularization layers like Batchnorm.
#          nn.BatchNorm2d after conv layers:
#          http://pytorch.org/docs/master/nn.html#batchnorm2d
#          nn.BatchNorm1d after fc layers:
#          http://pytorch.org/docs/master/nn.html#batchnorm1d
# This is a good resource for developing a CNN for classification:
# http://cs231n.github.io/convolutional-networks/#layers

import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        # TODO: define your model here
        self.conv2d1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3,padding=1)
        self.bn2d1 = nn.BatchNorm2d(num_features=256)
        self.relu1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,padding=1)
        self.bn2d2 = nn.BatchNorm2d(num_features=512)
        self.relu2 = nn.ReLU()
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=2)
        self.conv2d3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding=1)
        self.bn2d3 = nn.BatchNorm2d(num_features=512)
        self.relu3 = nn.ReLU()
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=2)
        self.conv2d4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,padding=1)
        self.bn2d4 = nn.BatchNorm2d(num_features=1024)
        self.relu4 = nn.ReLU()
        self.maxpool2d3 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2d4 = nn.MaxPool2d(kernel_size=4)
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1024, out_features=10)

        # base model:
        # self.conv2d1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # self.bn2d1 = nn.BatchNorm2d(num_features=6)   # BatchNorm2d
        # self.relu1 = nn.ReLU()
        # self.maxpool2d1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2d2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # self.bn2d2 = nn.BatchNorm2d(num_features=16)  # BatchNorm2d
        # self.relu2 = nn.ReLU()
        # self.maxpool2d2 = nn.MaxPool2d(kernel_size=2)
        # self.linear1 = nn.Linear(in_features=400, out_features=200)
        # #self.bn1d1 = nn.BatchNorm2d(num_features=200) # BatchNorm1d
        # self.relu3 = nn.ReLU()
        # self.linear2 = nn.Linear(in_features=200, out_features=10)

    def forward(self, x):

        # TODO: define your model here
        x = self.conv2d1(x)
        x = self.bn2d1(x)
        x = self.relu1(x)
        x = self.conv2d2(x)
        x = self.bn2d2(x)
        x = self.relu2(x)
        x = self.maxpool2d1(x)
        x = self.conv2d3(x)
        x = self.bn2d3(x)
        x = self.relu3(x)
        x = self.maxpool2d2(x)
        x = self.conv2d4(x)
        x = self.bn2d4(x)
        x = self.relu4(x)
        x = self.maxpool2d3(x)
        x = self.maxpool2d4(x)
        x = self.flat(x)
        x = self.linear1(x)

        # base model:
        # x = self.conv2d1(x)
        # x = self.bn2d1(x)
        # x = self.relu1(x)
        # x = self.maxpool2d1(x)
        # x = self.conv2d2(x)
        # x = self.bn2d2(x)
        # x = self.relu2(x)
        # x = self.maxpool2d2(x)
        # x = x.view(-1, 400)
        # x = self.linear1(x)
        # x = self.relu3(x)
        # x = self.linear2(x)

        return x

# Create an instance of the nn.module class defined above:
net = BaseNet()

# Test your BaseNet with some random input
dummy_input = torch.rand((1, 3, 32, 32))
output = net(dummy_input)
assert output.shape == torch.Size([1, 10])

# For training on GPU, we need to transfer net and data onto the GPU
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
if IS_GPU:
    net = net.cuda()

# TODO: paste output in your report
# run this and include the result in my PDF
print(net)

"""4.**Define a loss function and optimizer**"""

########################################################################
# Here we use Cross-Entropy loss and SGD with momentum.
# The CrossEntropyLoss criterion already includes softmax within its
# implementation. That's why we don't use a softmax in our model
# definition.

import torch.optim as optim
criterion = nn.CrossEntropyLoss()

# Tune the learning rate.
# See whether the momentum and weight decay is useful or not
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0)

"""5.**Train the model**"""

########################################################################
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize. We evaluate the validation accuracy at each
# epoch and plot these values over the number of epochs
# Nothing to change here
# -----------------------------
plt.ioff()
fig = plt.figure()
train_loss_over_epochs = []
val_accuracy_over_epochs = []

for epoch in tqdm(range(EPOCHS), total=EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        if IS_GPU:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    # Normalizing the loss by the total number of train batches
    running_loss/=len(trainloader)
    print('[%d] loss: %.3f' %
          (epoch + 1, running_loss))

    # Scale of 0.0 to 100.0
    # Calculate validation set accuracy of the existing model
    val_accuracy, val_classwise_accuracy = \
        calculate_accuracy(valloader, net, IS_GPU)
    print('Accuracy of the network on the val images: %d %%' % (val_accuracy))

    # # Optionally print classwise accuracies
    # for c_i in range(TOTAL_CLASSES):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[c_i], 100 * val_classwise_accuracy[c_i]))

    train_loss_over_epochs.append(running_loss)
    val_accuracy_over_epochs.append(val_accuracy.cpu())
# -----------------------------


# Plot train loss over epochs and val set accuracy over epochs
# Nothing to change here
# -------------
plt.subplot(2, 1, 1)
plt.ylabel('Train loss')
plt.plot(np.arange(EPOCHS), train_loss_over_epochs, 'k-')
plt.title('train loss and val accuracy')
plt.xticks(np.arange(EPOCHS, dtype=int))
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.arange(EPOCHS), val_accuracy_over_epochs, 'b-')
plt.ylabel('Val accuracy')
plt.xlabel('Epochs')
plt.xticks(np.arange(EPOCHS, dtype=int))
plt.grid(True)
plt.savefig("mp4_q1_plot.png")
plt.close(fig)
print('Finished Training')
# -------------

"""6.**Evaluate the validation accuracy of your final model**"""

val_accuracy, val_classwise_accuracy = \
        calculate_accuracy(valloader, net, IS_GPU)
print('Accuracy of the final network on the val images: %.1f %%' % (val_accuracy))

# Optionally print classwise accuracies
for c_i in range(TOTAL_CLASSES):
    print('Accuracy of %5s : %.1f %%' % (
        classes[c_i], val_classwise_accuracy[c_i]))

"""7.**Visualize test set images**"""

# get some random training images
dataiter = iter(testloader)
images = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images[:16]))

"""8.**Evaluate your final model on the test set**

Submit `predictions.npy` to Gradescope to see your model's performance on the test set.
"""

# run inference on the test set
predictions = run_secret_test(testloader, net, IS_GPU)
# save predictions
predictions = np.asarray(predictions)
np.save("Q1_label_predictions.npy", predictions)

"""#**Q2: Surface normal estimation**

Download the data (taskonomy_resize_128_release.zip) from google drive using this link (https://drive.google.com/file/d/1Y0ikK7f4-C3WYqi6UcdjRI2gwlU3pGGH/view?usp=sharing) and upload it to colab.
"""

!unzip -qqo /content/taskonomy_resize_128_release.zip



import glob
import os
import numpy as np
import random
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import models
from torchvision.transforms import ToTensor, Normalize

# global variable
device = torch.device("cuda:0")

class NormalDataset(data.Dataset):
    """
    Data loader for the Suface Normal Dataset. If data loading is a bottleneck,
    you may want to optimize this in for faster training. Possibilities include
    pre-loading all images and annotations into memory before training, so as
    to limit delays due to disk reads.
    """
    def __init__(self, split="train", data_dir="./taskonomy_resize_128_release"):
        assert(split in ["train", "val"])
        split2name = {
            "train": "allensville",
            "val": "beechwood",
        }
        self.img_dir = os.path.join(data_dir, split2name[split] + "_rgb")
        self.gt_dir = os.path.join(data_dir, split2name[split] + "_normal")

        self.split = split
        self.filenames = [
            os.path.splitext(os.path.basename(l))[0] for l in glob.glob(self.img_dir + "/*.png")
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(os.path.join(self.img_dir, filename) + ".png")
        img = np.asarray(img).copy()
        gt = Image.open(os.path.join(self.gt_dir, filename.replace("_rgb", "_normal")) + ".png")
        gt = np.asarray(gt)

        # from rgb image to surface normal
        gt = gt.astype(np.float32) / 255
        gt = torch.Tensor(np.asarray(gt).copy()).permute((2, 0, 1))
        mask = self.build_mask(gt).to(torch.float)

        img = ToTensor()(img)
        img = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        # normalize gt
        gt = gt * 2 - 1

        return img.contiguous(), gt, mask.sum(dim=0) > 0

    @staticmethod
    def build_mask(target, val=0.502, tol=1e-3):
        target = target.unsqueeze(0)
        if target.shape[1] == 1:
            mask = ((target >= val - tol) & (target <= val + tol))
            mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
            return (~mask).expand_as(target).squeeze(0)

        mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
        mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <= val + tol)
        mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <= val + tol)
        mask = (mask1 & mask2 & mask3).unsqueeze(1)
        mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
        return (~mask).expand_as(target).squeeze(0)

##########
#TODO: design your own network here. The expectation is to write from scratch. But it's okay to get some inspiration
#from conference paper. The bottom line is that you will not just copy code from other repo
##########
class MyModel(nn.Module):

    def __init__(self): # feel free to modify input paramters
        super(MyModel, self).__init__()
        # self.resnet18 = models.resnet18(pretrained=True)
        # self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-2]))

        # self.cv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=32, kernel_size=7, padding=3),
        #     #nn.BatchNorm2d(num_features=32),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, padding=3),
        #     #nn.BatchNorm2d(num_features=32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        # self.cv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        #     #nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     #nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        # self.cv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=128),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )

        # self.upsample2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # )
        # self.upsample2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # )
        # self.upsample2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # )
        # self.upsample1 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        # )
         # Encoder
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Decoder
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # upsample by factor of 2 8*8
        # self.conv11 = nn.Conv2d(256, 512, kernel_size=3, stride=3, padding=1)
        # self.rl1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        # self.conv22 = nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=1)
        # self.rl2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.conv33 = nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=1)
        # self.rl3 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.conv44 = nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=1)
        # self.rl4 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.conv55 = nn.Conv2d(128, 128, kernel_size=3, stride=3, padding=1)
        # self.rl5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 3, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder
        x_resnet = self.resnet18(x)

        x_d1 = self.down1(x) # 64
        x_d2 = self.down2(x_d1) # 32
        x_d3 = self.down3(x_d2) # 16
        x_d4 = self.down4(x_d3) # 8 256 8 8


        # Decoder
        x1 = self.conv1(x_resnet) # 8 256 8 8
        # x1 = self.conv11(x1)      # 8 512 2 2
        # print(x1.size())
        # x1 = self.rl1(x1)
        x = torch.cat([x1, x_d4], dim=1)
        x2 = self.conv2(x) # 16
        # x2 = self.conv22(x2)
        # x2 = self.rl2(x2)
        x = torch.cat([x2, x_d3], dim=1)
        x3 = self.conv3(x) # 32
        # x3 = self.conv33(x3)
        # x3 = self.rl3(x3)
        x = torch.cat([x3, x_d2], dim=1)
        x4 = self.conv4(x) # 64
        # x4 = self.conv44(x4)
        # x4 = self.rl4(x4)
        x = torch.cat([x4, x_d1], dim=1)
        x5 = self.conv5(x) # 128
        # x5 = self.conv55(x5)
        # x5 = self.rl5(x5)
#         print(x5.size())
        x = self.conv6(x5)

#         print(x.size())
        return x

    # def forward(self, x):
    #     x = self.resnet18(x)
    #     cv1 = self.cv1(x)     # 8 32 2 2
    #     # print(cv1.size())
    #     cv2 = self.cv2(cv1)   # 8 64 1 1
    #     # print(cv2.size())
    #     # cv3 = self.cv3(cv2)
    #     # print(cv3.size())

    #     # upsample3 = self.upsample3(cv3)
    #     # # print(upsample3.size())
    #     upsample2 = self.upsample2(cv2) # 8 32 2 2
    #     # upsample2 = self.upsample2(torch.cat([upsample3, cv2], dim=1))
    #     print(upsample2.size())
    #     upsample1 = self.upsample1(torch.cat([upsample2, cv1], dim=1))
    #     # print(upsample1.size())

    #     return upsample2

from google.colab import drive
drive.mount('/content/drive')

##########
#TODO: define your loss function here
##########
class MyCriterion(nn.Module):
    def __init__(self):
        super(MyCriterion, self).__init__()


    def forward(self, prediction, target, mask):
        loss = (F.l1_loss(prediction, target) * mask.unsqueeze(1)).mean()
        return loss

def simple_train(model, criterion, optimizer, train_dataloader, epoch, **kwargs):
    model.train()
    # TODO: implement your train loop here
    running_loss = 0
    for inputs, labels, mask in train_dataloader:
      inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels, mask)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      running_loss /= len(train_dataloader)
      print('[%d] loss: %.3f' %
          (epoch + 1, running_loss))
      val_accuracy, val_classwise_accuracy = \
        calculate_accuracy(valloader, net, IS_GPU)
      print('Accuracy of the network on the val images: %d %%' % (val_accuracy))

def angle_error(prediction, target):
    prediction_error = torch.cosine_similarity(prediction, target)
    prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
    prediction_error = torch.acos(prediction_error) * 180.0 / np.pi
    return prediction_error

def simple_predict(split, model):
    model.eval()
    dataset = NormalDataset(split=split)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=2, drop_last=False)
    gts, preds, losses = [], [], []
    total_normal_errors = None
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            img, gt, mask = batch
            img = img.to(device)
            gt = gt.to(device)
            mask = mask.to(device)

            pred = model(img)
            loss = (F.l1_loss(pred, gt, reduction="none") * mask.unsqueeze(1)).mean()

            gts.append((gt[0].permute((1, 2, 0)).cpu().numpy() + 1) / 2)
            preds.append((pred[0].permute((1, 2, 0)).cpu().numpy() + 1) / 2)
            losses.append(loss.item())

            angle_error_prediction = angle_error(pred, gt)
            angle_error_prediction = angle_error_prediction[mask > 0].view(-1)
            if total_normal_errors is None:
                total_normal_errors = angle_error_prediction.cpu().numpy()
            else:
                total_normal_errors = np.concatenate(
                    (total_normal_errors, angle_error_prediction.cpu().numpy())
                )

    return gts, preds, losses, total_normal_errors

########################################################################
# TODO: Implement your training cycles, make sure you evaluate on validation
# dataset and compute evaluation metrics every so often.
# You may also want to save models that perform well.

model = MyModel().to(device)
criterion = MyCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
train_dataset = NormalDataset(split='train')
train_dataloader = data.DataLoader(train_dataset, batch_size=8,
                                    shuffle=True, num_workers=2,
                                    drop_last=True)

num_epochs = 50
for epoch in range(num_epochs):
    simple_train(model, criterion, optimizer, train_dataloader, epoch)
    # consider reducing learning rate

"""# You do not need to change anything below"""

########################################################################
# Evaluate your result, and report
# 1. Mean angular error
# 2. Median angular error
# 3. Accuracy at 11.25 degree
# 4. Accuracy at 22.5 degree
# 5. Accuracy at 30 degree
# using provided `simple_predict` function.

def angle_error(prediction, target):
    prediction_error = torch.cosine_similarity(prediction, target)
    prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
    prediction_error = torch.acos(prediction_error) * 180.0 / np.pi
    return prediction_error

def simple_predict(split, model):
    model.eval()
    dataset = NormalDataset(split=split)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=2, drop_last=False)
    gts, preds, losses = [], [], []
    total_normal_errors = None
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            img, gt, mask = batch
            img = img.to(device)
            gt = gt.to(device)
            mask = mask.to(device)

            pred = model(img)
            loss = (F.l1_loss(pred, gt, reduction="none") * mask.unsqueeze(1)).mean()

            gts.append((gt[0].permute((1, 2, 0)).cpu().numpy() + 1) / 2)
            preds.append((pred[0].permute((1, 2, 0)).cpu().numpy() + 1) / 2)
            losses.append(loss.item())

            angle_error_prediction = angle_error(pred, gt)
            angle_error_prediction = angle_error_prediction[mask > 0].view(-1)
            if total_normal_errors is None:
                total_normal_errors = angle_error_prediction.cpu().numpy()
            else:
                total_normal_errors = np.concatenate(
                    (total_normal_errors, angle_error_prediction.cpu().numpy())
                )

    return gts, preds, losses, total_normal_errors

val_gts, val_preds, val_losses, val_total_normal_errors = simple_predict('val', model)
print("Validation loss (L1):", np.mean(val_losses))
print("Validation metrics: Mean %.1f, Median %.1f, 11.25deg %.1f, 22.5deg %.1f, 30deg %.1f" % (
    np.average(val_total_normal_errors), np.median(val_total_normal_errors),
    np.sum(val_total_normal_errors < 11.25) / val_total_normal_errors.shape[0] * 100,
    np.sum(val_total_normal_errors < 22.5) / val_total_normal_errors.shape[0] * 100,
    np.sum(val_total_normal_errors < 30) / val_total_normal_errors.shape[0] * 100
))

# vis validation
fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(22, 7))
for idx, ax_i in enumerate(axs.T):
    ax = ax_i[0]
    ax.imshow(val_gts[idx])
    ax.axis('off')
    ax = ax_i[1]
    ax.imshow(val_preds[idx])
    ax.axis('off')
fig.tight_layout()
plt.savefig('vis_valset.pdf', format='pdf', bbox_inches='tight')

# Visualization
# pick some of your favorite images and put them under `./data/normal_visualization/image`

class VisualizationDataset(data.Dataset):
    def __init__(self, image_dir="./taskonomy_resize_128_release", image_ext=".png"):
        self.img_dir = image_dir
        self.img_ext = image_ext

        self.img_dir = os.path.join(image_dir, "collierville_rgb")

        self.image_filenames = [
            os.path.splitext(os.path.basename(l))[0] for l in glob.glob(self.img_dir + "/*" + image_ext)
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        img = Image.open(os.path.join(self.img_dir, filename) + self.img_ext)
        img = np.asarray(img).copy()
        img = ToTensor()(img)
        img = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img.contiguous(), filename

def simple_vis(model):
    model.eval()
    dataset = VisualizationDataset(image_dir="./taskonomy_resize_128_release")
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=2, drop_last=False)
    imgs, preds = [], []

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            img, _ = batch
            img = img.to(device)

            pred = model(img)
            imgs.append(
                std * img[0].permute((1, 2, 0)).cpu().numpy() + mean
            )
            preds.append((pred[0].permute((1, 2, 0)).cpu().numpy() + 1) / 2)

    return imgs, preds

imgs, preds = simple_vis(model)
fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(22, 7))
for idx, ax_i in enumerate(axs.T):
    ax = ax_i[0]
    ax.imshow(imgs[idx])
    ax.axis('off')
    ax = ax_i[1]
    ax.imshow(preds[idx])
    ax.axis('off')
fig.tight_layout()
plt.savefig('q2_visualization.pdf', format='pdf', bbox_inches='tight')

# Test your model on the test set, submit the output to gradescope

from PIL import Image
import numpy as np

def simple_test(model, out_dir):
    model.eval()
    dataset = VisualizationDataset(image_dir="./taskonomy_resize_128_release")
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=2, drop_last=False)

    saved_predictions = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            img, filename = batch
            img = img.to(device)

            pred = model(img)
            saved_predictions.append(pred.cpu())

        saved_predictions = torch.cat(saved_predictions, dim=0)
        return saved_predictions

out_dir = "Q2_normal_predictions"
saved_predictions = simple_test(model, out_dir)
np.save('./Q2_surface_predictions.npy', saved_predictions)
