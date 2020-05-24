import glob

from tqdm import tqdm
import torch.utils.data as Data
from torch.autograd import Variable
import cv2

# from __future__ import print_function
# from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import torchvision.models as models
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score

def check_dataloader(dataloaders):
    def imshow(img, title=None):
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # Pause is necessary to display images correctly


    for inputs, labels in dataloaders["train"]:
        images = inputs.to(device)
        labels = labels.to(device)
        break
    grid_img = torchvision.utils.make_grid(images[:4], nrow=4)
    imshow(grid_img, title=[x for x in labels[:4]])
def save_net(model , PATH =  r"C:\Users\hxiao\PycharmProjects\gpugpugpu\hw4_deeplearning\alexmodel"):
    torch.save({
            'epoch': 100,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
            'loss': 0.2632,
             }, PATH)

def load_model():
    model.load_state_dict(torch.load(r'C:\Users\hxiao\PycharmProjects\gpugpugpu\hw4_deeplearning\alexmodel'))
    return model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=100, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #   Get model outputs and calculate loss
                    #   Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    model.cuda()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                no_improvement = 0


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)





            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Data augmentation and normalization for training
# Just normalization for validation

def load():
    train_dir = glob.glob(r"C:\Users\hxiao\PycharmProjects\gpugpugpu\hw4_deeplearning\train\**/*.png")
    # C:\Users\hxiao\PycharmProjects\gpugpugpu\hw4_deeplearning\train\Ariel_Sharon

    test_dir = glob.glob(r"C:\Users\hxiao\PycharmProjects\gpugpugpu\hw4_deeplearning\test\**/*.png")

    x_train = []
    y_train = []
    for dir in train_dir:
        y = dir[train_dir[0].find("ain") + 4:train_dir[0].find("ain") + 4 + 5]
        y_train.append(id.index(y))
        im = cv2.imread(dir)
        im2 = cv2.resize(im, (shape, shape), )
        x_train.append(im2)

    x_test = []
    y_test = []

    for dir in test_dir:
        y = dir[test_dir[0].find("est") + 4:test_dir[0].find("est") + 4 + 5]
        y_test.append(id.index(y))
        im = cv2.imread(dir)
        im2 = cv2.resize(im, (shape, shape), )
        x_test.append(im2)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train_dataset = Data.TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train)
    )

    test_dataset = Data.TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(y_test)
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

    datasets_sizes = {"train": 1320, "val": 240}
    use_gpu = torch.cuda.is_available()

    dataloaders = dict()
    dataloaders["train"] = train_loader
    dataloaders["val"] = test_loader

    return dataloaders, x_train,x_test,y_train,y_test

if __name__ == '__main__':#(224, 224, 3) X 1320
#https://pytorch.org/hub/pytorch_vision_alexnet/
    alexnet = models.alexnet(pretrained=False)


    vgg16 = models.vgg16(pretrained=False)
# dataloaders, x_train, x_test, y_train, y_test = load()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    batch_size = 16

    feature_extract = True
    shape = 224
    num_epochs = 100
    input_size = shape


    model = alexnet

    model = model.to(device)

    id = ["Ariel", "Colin", "Donal", "Georg", "Gerha", "Hugo_", "Jacqu", "Jean_", "John_", "Junic", "Seren", "Tony_"]
    num_classes = len(id)
    model.classifier[6] = nn.Linear(4096, num_classes)#change the final FC layer from 1000classes to 12 classes.
    conv_list = [0, 3, 6, 8, 10]
    fc_list = [1, 4, 6]

    for i in conv_list:
        nn.init.kaiming_normal_(model.features[i].weight)
    for i in fc_list:
        nn.init.kaiming_normal_(model.classifier[i].weight)

    model.features

    params_to_update = model.parameters()
    print("Params to learn:")

    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(r"C:\Users\hxiao\PycharmProjects\gpugpugpu\hw4_deeplearning", x),
                                data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
        ['train', 'val']}

    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    save_net(model, PATH=r"C:\Users\hxiao\PycharmProjects\gpugpugpu\hw4_deeplearning\alexmodel")

    ohist = []
    ohist = [h.cpu().numpy() for h in hist]
    plt.title("AlexNet Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1),ohist,label="Validation")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()

    # Iterate over data.
    #plot the ROC curve:


    model = alexnet
    optimizer = optimizer_ft

    checkpoint = torch.load(r"C:\Users\hxiao\PycharmProjects\gpugpugpu\hw4_deeplearning\alexmodel")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    a = []
    b = []
    cnt = 0
    for inputs, labels in dataloaders_dict["val"]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.cuda()
        labels = labels.cuda()

        model.cuda()
        outputs = model(inputs)
        y_score, preds = torch.max(outputs, 1)
        lst = preds == labels
        y_true = lst.cpu().numpy()
        y_score = y_score.cpu().detach().numpy()
        a.append(y_true)
        b.append(y_score)

    a = np.array(a)
    b = np.array(b)
    a = a.flatten()
    b = b.flatten()
    y_true = a
    y_score = b
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    roc_auc = auc(y_true, y_score)

    # Plot ROC curve
    # ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(y_true, y_score)
    # summarize scores
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_score)
    # plot the roc curve for the model
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

