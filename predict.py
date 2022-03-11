#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Shankary Ravichelvam
# DATE CREATED: 11/03/2022
# REVISED DATE: 
# PURPOSE: To predict/classify image from trained model
 
# All necessary imports of packages to be used
import argparse
import sys

import matplotlib.pyplot as plt

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
parser.add_argument('--save_dir', dest='save_dir', type = str, default = './checkpoint.pth', help='Set directory to save checkpoints')
parser.add_argument('--image_path', type = str, default = 'flowers/test/10/image_07090.jpg', help='Path to image for processing')
parser.add_argument('--topk', type = int, default = 5, help = 'Top K classes')
    
args = parser.parse_args()


def main():
    # Using GPU if available
    gpu = args.gpu
    if gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    # Loading the checkpoint and rebuilding the model using the user's input
    save_dir = args.save_dir
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        input_size = checkpoint['input_size']
        output_size = checkpoint['output_size']
        hidden_layers = checkpoint['hidden_layers']
        epochs = checkpoint['epochs']
        learn_rate = checkpoint['learn_rate']
        print_every = checkpoint['print_every']
        
        return model
    
    model = load_checkpoint(save_dir)
    print(model)
    
    #Image Preprocessing
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        '''
        
        # Processing a PIL image for use in a PyTorch model
        image = Image.open(image) # Opening image using PIL
        
        # Resize image and crop
        process_image = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
        image = process_image(image)
        
        return image
    
    def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
            
            # PyTorch tensors assume the color channel is the first dimension
            # but matplotlib assumes is the third dimension
            image = image.numpy().transpose((1, 2, 0))
            
            # Undo preprocessing
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            
            # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
            image = np.clip(image, 0, 1)
            
            ax.imshow(image)
            
            return ax
    
    # Class Prediction
    def predict(image_path, model, topk=5):
        ''' 
        Predict the class (or classes) of an image using a trained deep learning model.
        '''
        
        image = process_image(image_path).type(torch.FloatTensor).unsqueeze_(0)
        
        load_checkpoint(save_dir)
        
        model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
        
        with torch.no_grad():
            outputs = model(image.to('cuda'))
            ps = torch.exp(outputs)
            probs, indices = ps.topk(args.topk)
            probs = probs.squeeze() # Used the helper function from previous workspace and Knowledge section to help
            classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
            
        return probs, classes
    
    # Classifying the image
    image_path = args.image_path
    flower_img_number = image_path.split('/')[2]
    flower_name = cat_to_name[flower_img_number]
    top_probs, top_classes = predict(image_path, model, args.topk)
    class_names = [cat_to_name[c1] for c1 in classes]
    
    print(top_probs)
    print(class_names)

if __name__ == "__main__":
    main()