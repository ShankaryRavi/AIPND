#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Shankary Ravichelvam
# DATE CREATED: 10/03/2022
# REVISED DATE: 
# PURPOSE: To retrieve command line inputs from user to train model
 
# All necessary imports of packages to be used
import argparse
import sys

import torch
from torchvision import datasets, transforms, models
    
from torch import nn, optim
import torch.nn.functional as F
    
import PIL
from PIL import Image
    
import numpy as np
    
import json
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Obtaining user input from CLI
parser = argparse.ArgumentParser()
    
parser.add_argument('--gpu', action = 'store_true', default=True, help= 'Do you want to use GPU?')
parser.add_argument('--data_dir', type = str, default = './flowers', help= 'Location of flower images to train model')
parser.add_argument('--arch', type = str, default = models.vgg16(pretrained=True), help='CNN Model architecture to use. Recommended: VGG16')
parser.add_argument('--save_dir', dest='save_dir', type = str, default = './checkpoint.pth', help='Set directory to save checkpoints')
parser.add_argument('--learning_rate', type = float, default = 0.001, help= 'Set learning rate parameter')
parser.add_argument('--input_size', type = int, action= 'store', default=25088, dest='input_size', help='Input size')
parser.add_argument('--hidden_layer1', type = int, action='store', default = 4096, dest='hidden_layer1', help='Number of hidden units in layer 1:')
parser.add_argument('--hidden_layer2', type = int, default = 512, help='Number of hidden units in layer 2:')
parser.add_argument('--output_size', type=int, action='store', default=102, dest='output_size', help='Output size, i.e. no. of classes of flowers')
parser.add_argument('--epochs', type = int, default = 5, help='Set epochs between 1 and 10.')
    
args = parser.parse_args()


def main():
                      
    data_dir = args.data_dir
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'
    
    # Defining transforms for training, validation and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Using the image datasets and the transforms to define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    # Using model as defined by user, default is vgg16
    model = args.arch
    
    # Defining new, untrained feed-forward network as a classifier
    
    # Using GPU if available
    gpu = args.gpu
    if gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
   
    
    # Freezing parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replacing hardcoded values for user's input
    model.classifier = nn.Sequential(nn.Linear(args.input_size, args.hidden_layer1),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(args.hidden_layer1, args.hidden_layer2),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(args.hidden_layer2, args.output_size),
                                     nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    
    # Training classifier parameters, feature parameters are frozen. User's input for learning rate
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    
    model = model.to(device)
    
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    from workspace_utils import active_session
    
    with active_session():
        epochs = args.epochs # User's input here
        steps = 0
        running_loss = 0
        print_every = 50
        
        for e in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            
                            valid_loss += batch_loss.item()
                            
                            # Calculating the accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                    # Tracking loss and accuracy on the validation set to determine the best hyperparameters
                    print(f"Epoch {e+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                          f"Valid accuracy: {accuracy/len(validloader):.3f}")
                    
                    running_loss = 0
                    model.train()
     
    # Doing validation on the test set
    def test_network(testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images.to('cuda'))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to('cuda')).sum().item()
                
        print('Accuracy of the network on the test images: %d %%' % (100* correct / total))
    
    test_network(testloader)
    
    # Saving the checkpoint
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {'arch': args.arch,
                  'input_size': args.input_size,
                  'output_size': args.output_size,
                  'hidden_layers': [args.hidden_layer1, args.hidden_layer2],
                  'optimizer': optimizer.state_dict(),
                  'epochs': args.epochs,
                  'learn_rate': args.learning_rate,
                  'print_every': 50,
                  'state_dict': model.state_dict()}
    
    torch.save(checkpoint, 'checkpoint.pth')

if __name__ == "__main__":
    main()