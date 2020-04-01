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
parser = argparse.ArgumentParser(description="Predict")
parser.add_argument('input', default="./flowers/test/1/image_06752.jpg", type=str, help='Image Input')
parser.add_argument('checkpoint', default='./checkpoint.pth', type = str, help="Checkpoint file")
parser.add_argument('--top_k', default=3, type=int, help='Provide topk')

args = parser.parse_args()

image = args.input
top_k = args.top_k
checp = args.checkpoint

def main():
    model = uf.loadModel(checp)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
        
    pts = uf.predict(image, model, top_k)
    labels = [cat_to_name[str(index + 1)] for index in np.array(pts[1][0])]
    probability = np.array(pts[0][0])
    i=0
    while i < top_k:
        print("{} with a probability of {:.3f}".format(labels[i], probability[i]))
        i += 1
    
    print("Done Predicting!")

    
if __name__== "__main__":
    main()
