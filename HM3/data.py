# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:14:56 2019

@author: Aicha BOUJANDAR
"""

import zipfile
import os

import torchvision.transforms as transforms

##We apply different transformations to our training, validation and test dataset in order to have varied images


data_transforms_train = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),                                                                   
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

data_transforms_val = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


data_transforms_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])