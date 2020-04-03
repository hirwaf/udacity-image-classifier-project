import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def dataloaders(dir_  = "./flowers" ):
    
    data_dir = dir_
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train' : transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229,0.224,0.255])]),

        'valid' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229,0.224,0.255])]),

        'test' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229,0.224,0.255])]),
    }
   

    # Load the datasets with ImageFolder ['train', 'valid', 'test']
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid' : datasets.ImageFolder(train_dir, transform=data_transforms['valid']),
        'test' : datasets.ImageFolder(train_dir, transform=data_transforms['test']),
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
        'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True),
    }
    
    return image_datasets, dataloaders

def nnNet(architecture='densenet121', h_layer=120, dp=0.5, lr=0.001):
    if architecture=='densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("\nAllowed architectures are densenet121 or vgg16.\n")
    
    for p in model.parameters():
        p.requireds_grad = False
    
    classifier = nn.Sequential(OrderedDict([       
        ('inputs', nn.Linear(1024, h_layer)),
        ('relu1', nn.ReLU()),
        ('h_layer1', nn.Linear(h_layer, 90)),
        ('relu2',nn.ReLU()),
        ('h_layer2',nn.Linear(90,80)),
        ('relu3',nn.ReLU()),
        ('h_layer3',nn.Linear(80,102)),
        ('dropout',nn.Dropout(dp)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    return model, optimizer, criterion

def trainNet(model, criterion, optimizer, dataloaders, epochs=6, gpu=True):   
    steps = 0
    loss = []
    print_every = 5
    
    if torch.cuda.is_available() and gpu:
        model.to('cuda')
    
    print("\n-------------- Start training -----------------------\n")

    for epoch in range(epochs):
        running_loss = 0

        for i, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1
            if torch.cuda.is_available() and gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                for ii, (inputs2, labels2) in enumerate(dataloaders['valid']):
                    optimizer.zero_grad()
                    
                    if torch.cuda.is_available() and gpu:
                        inputs2, labels2 = inputs2.to('cuda'), labels2.to('cuda')
                        model.to('cuda')

                    with torch.no_grad():
                        logps = model.forward(inputs2)
                        test_loss = criterion(logps, labels2)
                        ps = torch.exp(logps).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                test_loss = test_loss / len(dataloaders['valid'])
                accuracy = accuracy /len(dataloaders['valid'])

                print(f"Epoch: {(epoch+1)}/{epochs} ..",
                      f"Train Loss: {(running_loss/print_every):.3f} ..",
                      f"Valid Lost {test_loss:.3f} ..",
                      f"Accuracy: {accuracy:.3f}")


                running_loss = 0
                model.train()
    
    print("-------------- Finished training -----------------------")
    print(f'Epochs: {epochs}------------------------------------')
    print(f'Steps: {steps}-----------------------------')

def saveCheckpoint(model, train_class_to_idx, path='checkpoint.pth', arch='densenet121', lr=0.001, dp=0.5, epochs=6):
    model.class_to_idx = train_class_to_idx
    model.cpu
    torch.save({'structure' : arch,
            'h_layer':120,
            'dp':dp,
            'lr':lr,
            'epochs':epochs,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            path)

def loadModel(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    h_layer = checkpoint['h_layer']
    arch = checkpoint['structure']
    model,_,_ = nnNet(arch, h_layer)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def processImage(image):
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor

def predict(image_path, model, topk=5, gpu=True):
    if torch.cuda.is_available() and gpu == True:
        model.to('cuda:0')
        
    img_torch = processImage(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        if torch.cuda.is_available() and gpu == True:
            img_torch = img_torch.cuda()
            
        output = model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)
    
    
