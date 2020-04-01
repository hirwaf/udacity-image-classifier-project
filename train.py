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
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate during learning')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout during learning')
parser.add_argument('--hidden_units', type=int, default=120, help='list of number hidden layers')
parser.add_argument('--epochs', type=int, default=6, help='number of epochs')

args = parser.parse_args()

dir_ = args.data_dir
save_dir = args.save_dir
lr = args.learning_rate
dp = args.dropout
h_layer = args.hidden_units
epochs = args.epochs

def main():
    image_datasets, dataloaders = uf.dataloaders(dir_)
    model, optimizer, criterion = uf.nnNet(h_layer, dp, lr)
    uf.trainNet(model, criterion, optimizer, dataloaders, epochs)
    uf.saveCheckpoint(model, train_class_to_idx=image_datasets['train'].class_to_idx, dp=dp, lr=lr, epochs=epochs)
    
    print("\nTraining process is complete!!")

if __name__ == "__main__":
    main()





