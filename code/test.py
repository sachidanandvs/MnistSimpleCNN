# imports ---------------------------------------------------------------------#
import sys
import os
import argparse
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
from ema import EMA
from datasets import MnistDataset
from transforms import RandomRotation
from models.modelM3 import ModelM3
from models.modelM5 import ModelM5
from models.modelM7 import ModelM7

import matplotlib.pyplot as plt
import cv2
import random


def get_position(truncation,o_w,o_h):
    tmax = truncation[1]/100
    tmin = truncation[0]/100
    w, h = 28, 28
    val = random.random()
    if(o_w > 10):
        if(val < 0.25):
            x = random.randint(
                int(-(tmax) * o_w),
                int(-(tmin) * o_w),
            )
            y = random.randint(0, h - o_h)
        elif(0.25 <= val < 0.5):
            x = random.randint(
                w - o_w + int((tmin) * o_w),
                w - o_w + int((tmax) * o_w),
            )
            y = random.randint(0, h - o_h)
        elif(0.5 <= val < 0.75):
            y = random.randint(
                int(-(tmax) * o_h),
                int(-(tmin) * o_h),
            )
            x = random.randint(0, w - o_w)
        else:
            y = random.randint(
                h - o_h + int((tmin) * o_h),
                h - o_h + int((tmax) * o_h),
            )
            x = random.randint(0, w - o_w)
    else:
        if(val < 0.5):
            y = random.randint(
                int(-(tmax) * o_h),
                int(-(tmin) * o_h),
            )
            x = random.randint(0, w - o_w)
        else:
            y = random.randint(
                h - o_h + int((tmin) * o_h),
                h - o_h + int((tmax) * o_h),
            )
            x = random.randint(0, w - o_w)
    return x,y

def shift_image(image,tr):
    image = image.cpu().numpy().squeeze(0)
    W, H = image.shape
    # get bounding box around white pixels in image
    ori_img = image.copy()*255
    bimg = image.copy()
    bimg = ((bimg)*255).astype(np.uint8)
    bimg = cv2.adaptiveThreshold(bimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)

    # cont = cv2.findNonZero(bimg)
    # x, y, w, h = cv2.boundingRect(cont)
    # cv2.rectangle(bimg, (x, y), (x+w, y+h), (255, 255, 255),2)
    # cv2.imshow("image",bimg)
    # cv2.waitKey(0)
    
    cont = cv2.findNonZero(bimg)
    x1, y1, w1, h1 = cv2.boundingRect(cont)

    x_l, y_l = get_position(tr,w1,h1)
    background = np.zeros((3*W, 3*H), np.uint8)
    background[W + x_l:W+x_l+w1, H+y_l:H+y_l+h1] = ori_img[x1:x1+w1,y1:y1+h1]
    cv2.imshow("image",background[W:2*W,H:2*H])
    cv2.waitKey(0)
    return background[W:2*W,H:2*H]/255

def run(p_seed=0, p_kernel_size=5, p_logdir="temp"):

    # enable GPU usage ------------------------------------------------------------#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")
        exit(0)

    # data loader -----------------------------------------------------------------#
    test_dataset = MnistDataset(training=False, transform=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    # model selection -------------------------------------------------------------#
    if(p_kernel_size == 3):
        model1 = ModelM3().to(device)
    elif(p_kernel_size == 5):
        model1 = ModelM5().to(device)
    elif(p_kernel_size == 7):
        model1 = ModelM7().to(device)

    model1.load_state_dict(torch.load("../logs/%s/model%03d.pth"%(p_logdir,p_seed)))

    # Truncation
    truc = [10,25]

    model1.eval()
    test_loss = 0
    correct = 0
    max_correct = 0
    total_pred = np.zeros(0)
    total_target = np.zeros(0)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device,  dtype=torch.int64)
            print(data.shape)
            new_input = []
            for i in range(data.shape[0]):
                new_input.append(shift_image(data[i],truc))
            
            data = torch.from_numpy(np.array(new_input)).view(data.shape[0],1,28,28).to(device).float()
            output = model1(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            # total_pred = np.append(total_pred, pred.cpu().numpy())
            # total_target = np.append(total_target, target.cpu().numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        if(max_correct < correct):
            max_correct = correct
            print("Best accuracy! correct images: %5d"%correct)

    
    plt.imshow(data[0].cpu().numpy().reshape(28,28), cmap='gray')
    plt.show()

    #--------------------------------------------------------------------------#
    # output                                                                   #
    #--------------------------------------------------------------------------#
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / len(test_loader.dataset)
    best_test_accuracy = 100 * max_correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) (best: {:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy, best_test_accuracy))



    # test_loss = 0
    # correct = 0
    # wrong_images = []
    # with torch.no_grad():
    #     for batch_idx, (data, target) in enumerate(test_loader):
    #         data, target = data.to(device), target.to(device)
    #         print(data[0])
    #         output = model1(data)
    #         test_loss += F.nll_loss(output, target, reduction='sum').item()
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #         wrong_images.extend(np.nonzero(~pred.eq(target.view_as(pred)).cpu().numpy())[0]+(100*batch_idx))

    # np.savetxt("../logs/%s/wrong%03d.txt"%(p_logdir,p_seed), wrong_images, fmt="%d")
    # #print(len(wrong_images), wrong_images)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="modelM5")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--trials", default=30, type=int)
    p.add_argument("--kernel_size", default=5, type=int)
    args = p.parse_args()
    for i in range(args.trials):
        run(p_seed = args.seed + i,
            p_kernel_size = args.kernel_size,
            p_logdir = args.logdir)




