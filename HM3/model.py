# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:14:11 2019

@author: Aicha BOUJANDAR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models #the pretrained models in Pytorch

nclasses = 20 

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, nclasses)
for name, child in model.named_children():
   if name in ['fc','layer3','layer4']:
       print(name + ' is unfrozen')
       for param in child.parameters():
           param.requires_grad = True
   else:
       print(name + ' is frozen')
       for param in child.parameters():
           param.requires_grad = False