import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import json
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import functions as uf

# Set up argument parser
parser = argparse.ArgumentParser(description="Training Network")
parser.add_argument('data_dir', default="./flowers", help='directory containing data')
parser.add_argument('--save_dir', default="./checkpoint.pth", help='directory for saving checkpoint')
parser.add_argument('--architecture', type=str, default="densenet121", help='allowed architecture: densenet121 or vgg16')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate during learning')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout during learning')
parser.add_argument('--hidden_units', type=int, default=120, help='list of number hidden layers')
parser.add_argument('--epochs', type=int, default=6, help='number of epochs')
parser.add_argument('--gpu', type=int, default=1, help='use gpu or cpu')

args = parser.parse_args()

dir_ = args.data_dir
save_dir = args.save_dir
lr = args.learning_rate
dp = args.dropout
h_layer = args.hidden_units
epochs = args.epochs
arch = args.architecture
gpu = True if args.gpu == 1 and torch.cuda.is_available() else False

def main():
    image_datasets, dataloaders = uf.dataloaders(dir_)
    model, optimizer, criterion = uf.nnNet(arch, h_layer, dp, lr)
    uf.trainNet(model, criterion, optimizer, dataloaders, epochs, gpu)
    uf.saveCheckpoint(model, train_class_to_idx=image_datasets['train'].class_to_idx, arch=arch, dp=dp, lr=lr, epochs=epochs)
    
    print("\nTraining process is complete!!")

if __name__ == "__main__":
    main()





