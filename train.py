from __future__ import print_function
from __future__ import division
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
from models import EfficientNet


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', metavar='DIR',default="./data_256_stacked",
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                    help='model architecture (default: resnet101)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-n_classes',default=1500, type=int, help='number of classses')


def get_dataloaders(data_dir, input_size):
    print("Initializing Datasets and Dataloaders...")

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Pad((0, (192) // 2)),
            transforms.Resize(input_size),
            transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Pad((0, (192) // 2)),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    return {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}


def initialize_model(model_name, num_classes, learning_rate, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = 0
    optimizer = None
    criterion = None

    if model_name == "resnet101":
        """ Resnet101
        """
        model = models.resnet101(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

    elif model_name == "resnet152":
        """ Resnet101
        """
        model = models.resnet152(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

    elif model_name == "resnet101-next":
        """ Resnet101
        """
        model = models.resnext101_32x8d(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

    elif model_name == "resnet50-next":
        """ Resnet101
        """
        model = models.resnext50_32x4d(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

    elif model_name == "wide-resnet101-2":
        """ Resnet101
        """
        model = models.wide_resnet101_2(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()


    elif model_name == "densenet161":
        """ Densenet
        """
        model = models.densenet161(pretrained=use_pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

    elif model_name == "densenet201":
        """ Densenet
        """
        model = models.densenet201(pretrained=use_pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs , num_classes)
        input_size = 224
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

    elif model_name == "mobilenet":
        """ MobileNet
        """
        model = models.mobilenet_v2(pretrained=use_pretrained)
        # num_ftrs = model.classifier.in_features
        # model.classifier = nn.Linear(num_ftrs, num_classes)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        input_size = 224
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
        params_to_update = model.parameters()
        optimizer = optim.Adam(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

    elif model_name == "efficientnet-b3":
        model = EfficientNet.from_pretrained(model_name)
        in_ftrs = model._fc.in_features
        model._fc = nn.Linear(in_ftrs, num_classes)
        input_size = 224
        params_to_update = model.parameters()
        optimizer = optim.RMSprop(params_to_update, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, optimizer, criterion, input_size


def train_model(model_name, model, dataloaders, criterion, optimizer, device, learning_rate, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_runtimes_history = []

    for epoch in range(num_epochs):
        if epoch != 0:
            endtime_estimation = (num_epochs - epoch) * np.mean(epoch_runtimes_history)
        print(f'Epoch {epoch}/{num_epochs - 1} - estimated time left: {"starting..." if len(epoch_runtimes_history) == 0 else str(datetime.timedelta(seconds=endtime_estimation))}')
        print('-' * 10)

        epoch_start_time = time.time()

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

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if model_name == "inception" and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == "val":
                print('best val acc: {:4f}'.format(best_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),
                           f"{model_name}_e_{num_epochs}_lr_{str(learning_rate).replace('.','_')}_best.pth")

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        epoch_runtime = time.time() - epoch_start_time
        epoch_runtimes_history.append(epoch_runtime)
        print(f"Epoch runtime: {str(datetime.timedelta(seconds=epoch_runtime))}")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == '__main__':
    args = parser.parse_args()

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = args.data

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = args.arch

    # Number of classes in the dataset
    num_classes = args.n_classes

    # Batch size for training (change depending on how much memory you have)
    batch_size = args.batch_size

    # Number of epochs to train for
    num_epochs = args.epochs

    learning_rate = args.lr

    # # Override
    # data_dir = "./data_256_stacked"
    # model_name = "inception"
    # num_classes = 25
    # batch_size = 16
    # num_epochs = 40

    model, optimizer, criterion, input_size = initialize_model(model_name, num_classes, learning_rate, use_pretrained=True)

    # Print the model we just instantiated
    print(model)

    # Create training and validation dataloaders
    dataloaders_dict = get_dataloaders(data_dir, input_size)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model = model.to(device)

    # Train and evaluate
    model_ft, hist = train_model(model_name, model, dataloaders_dict, criterion, optimizer, device, learning_rate, num_epochs=num_epochs)

