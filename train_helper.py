# Imports here
import torch
import torchvision
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy
import torch.nn.functional as F

models_dict = { 'resnet18' : models.resnet18(pretrained=True),
                'alexnet' : models.alexnet(pretrained=True),
                'squeezenet' : models.squeezenet1_0(pretrained=True),
                'vgg16' : models.vgg16(pretrained=True),
               'vgg13' : models.vgg13(pretrained=True),
               'resnet50' : models.resnet50(pretrained=True)
            
}


def preprocess_and_train(args_dict):
    dataloaders, model, optimizer, criterion, image_datasets = preprocess(args_dict['data_dir'], args_dict['arch'], args_dict['learning_rate'], args_dict['gpu'])
    
    train(args_dict['epochs'], dataloaders, model, optimizer, criterion, args_dict['gpu'], args_dict['data_dir'] + args_dict['save_dir'], image_datasets)
    


def preprocess(data_dir_in, arch, learning_rate, gpu):
    
    data_dir = data_dir_in + 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    num_workers = 0
    batch_size = 20

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(224), 
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

    image_datasets = {'train': train_data,'valid' : valid_data, 'test': test_data}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
        num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers, shuffle=True)

    dataloaders = {'train':train_loader,
                      'valid':valid_loader,
                      'test':test_loader}

    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    #Build your network
    use_cuda = gpu #torch.cuda.is_available()
    # Load the pretrained model from pytorch
    model = models_dict[arch]

    # Freeze training for all layers
    for param in model.parameters():
        param.requires_grad = False

    # Update the last layer to create the desired classifier
    if arch == "vgg13" or arch == "vgg16":        
        # Unfreeze only the classifier
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 102)
    else:        
        # Unfreeze only the classifier
        for param in model.fc.parameters():
            param.requires_grad = True
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 102)



    if use_cuda:
        model = model.cuda()

    # Print model
    print(model)

    criterion = nn.CrossEntropyLoss()
    if arch == "vgg13" or arch == "vgg16":
        optimizer = optim.Adam(model.classifier[-1].parameters(), lr = learning_rate)        
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr = learning_rate)

    return dataloaders, model, optimizer, criterion, image_datasets


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path, image_datasets):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    number_of_no_improvement = 0
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
        #for data, target in loaders['train']:
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            #output = model_scratch(data[batch_idx].unsqueeze_(0))
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            #train_loss += loss.item()*data.size(0)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        #for batch_idx, (data, target) in enumerate(loaders['valid']):
        for data, target in loaders['valid']:
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            
            # calculate the batch loss
            loss = criterion(output, target)
            
            # update average validation loss 
            #valid_loss += loss.item()*data.size(0)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        
        #train_loss = train_loss/len(train_loader.dataset)
        #valid_loss = valid_loss/len(valid_loader.dataset)
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        number_of_no_improvement +=1
        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            model.class_to_idx = image_datasets['train'].class_to_idx
            model.optimizer_state_dict = optimizer.state_dict
            torch.save(model.state_dict(), save_path)
            torch.save(model.class_to_idx, 'class_to_idx.pt')
            valid_loss_min = valid_loss
            number_of_no_improvement = 0
        
        # exit there is no improvement for 10 consecutive times
        if number_of_no_improvement >= 5:
            return model
    # return trained model
    return model