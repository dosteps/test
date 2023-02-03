import argparse
import os
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test

class_num = 4 #cat dog person background

num_epochs = 300
batch_size = 16


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
network.to(device)

cudnn.benchmark = True

mAP_best = 0


# split dataset into train/val
dataset_dir = "./data/train/"            #write the path to the directory of whole dataset(...../train/)
train_dir = "./train_set/"               #write a new data path in front of /train_set/,
val_dir = "./val_set/"                   # split_dataset function automatically makes a new folder
split_ratio = 0.9

split_dataset(dataset_dir, train_dir, val_dir, split_ratio)

# make a directory that saves precision recall curve
plt_dir = './precision_recall_curve/'
if os.path.exists(plt_dir):
    print(f"{plt_dir} exists, removing...")
    shutil.rmtree(plt_dir, ignore_errors=True)

os.makedirs(plt_dir, exist_ok=True)


if not args.test:
    imgs_dir = train_dir + 'images/'
    annot_dir = train_dir + 'annotations/'
    imgs_val_dir = val_dir + 'images/'
    annot_val_dir = val_dir + 'annotations/'
    dataset = COCO(imgs_dir, annot_dir, class_num, boxs_default, train=True, image_size=320)
    dataset_val = COCO(imgs_val_dir, annot_val_dir, class_num, boxs_default, train=False, image_size=320)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)          #shuffle -> True!!
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    print("Total Epoch:", num_epochs, ", Batch Size:", batch_size)

    for epoch in range(num_epochs):
        print("Epoch:", epoch)

        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        print("[", end='')
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, width, height = data
            images = images_.to(device)
            ann_box = ann_box_.to(device)
            ann_confidence = ann_confidence_.to(device)

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1


            if i % 10 == 9: print("#", end='')
        print("]")
        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))

        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("train_raw_" + str(epoch), pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        pred_confidence_, pred_box_ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default, overlap=0.35, threshold=0.5)
        visualize_pred("train_nms_" + str(epoch), pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

        # break               #Erase later!!!!!!!!!
        
        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        
        thresholds = np.arange(start=0, stop=0.95, step=0.05)

        ann_conf_dict = {0: [], 1: [], 2: []}
        pred_conf_dict = {0: [], 1: [], 2: []}

        for i, data in enumerate(dataloader_val, 0):
            images_, ann_box_, ann_confidence_, width, height = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
            # generate_mAP(ann_confidence_[0].numpy(), pred_confidence_[0], precisions, recalls, APs)
            ann_confidence = ann_confidence_.numpy()

            for (pred_data, ann_data) in zip(pred_confidence_, ann_confidence):
                label_pos = np.argwhere(ann_data[:, :-1] == 1)  # finding the indices of labels
                ann_num = label_pos.shape[0]  # finding the number of labels
                c = label_pos[0, 1]  # finding which class it is

                ann_conf = ann_data[:, c]
                for element in ann_conf:
                    ann_conf_dict[c].append(int(element))
                
                pred_conf = pred_data[:, c]
                for element in pred_conf:
                    pred_conf_dict[c].append(element)


        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        pred_confidence_, pred_box_ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default, overlap=0.35, threshold=0.5)
        visualize_pred("val_" + str(epoch), pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        mAP = generate_mAP(ann_conf_dict, pred_conf_dict, thresholds, epoch)
        print(f"mAP: {mAP:.3f}")
        
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        #save weights
        # save weights in every epoch
        torch.save(network.state_dict(), 'network.pth')

        if mAP >= mAP_best:
            print('New record! Updating network_best...')
            torch.save(network.state_dict(), 'network_best.pth')
            mAP_best = mAP


else:
    #TEST
    test_imgs_dir = val_dir + 'images'
    test_annot_dir = val_dir + 'annotations'
    dataset_test = COCO(test_imgs_dir, test_annot_dir, class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()

    txt_dir = './result/'
    if os.path.exists(txt_dir):
        print(f"{txt_dir} exists, removing...")
        shutil.rmtree(txt_dir, ignore_errors=True)

    os.makedirs(txt_dir, exist_ok=True)

    thresholds = np.arange(start=0, stop=0.95, step=0.05)

    ann_conf_dict = {0: [], 1: [], 2: []}
    pred_conf_dict = {0: [], 1: [], 2: []}
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, width, height = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)

        ann_confidence = ann_confidence_.numpy()

        for (pred_data, ann_data) in zip(pred_confidence_, ann_confidence):
            label_pos = np.argwhere(ann_data[:, :-1] == 1)  # finding the indices of labels
            ann_num = label_pos.shape[0]  # finding the number of labels
            c = label_pos[0, 1]  # finding which class it is

            ann_conf = ann_data[:, c]
            for element in ann_conf:
                ann_conf_dict[c].append(int(element))
            
            pred_conf = pred_data[:, c]
            for element in pred_conf:
                pred_conf_dict[c].append(element)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment

        txt_dir = "./result/"                    #write your own directory where you will save your test results
        filename = f'test_{str(i).zfill(5)}.txt'

        width = int(width)
        height = int(height)

        with open(txt_dir + filename, 'w') as file:
            for idx in range(len(pred_box_)):
                if pred_box_[idx].all() == 0:
                    continue

                tx, ty, tw, th = pred_box_[idx, :]
                px, py, pw, ph = boxs_default[idx, :4]

                gx = (pw * tx + px) * width  # getting actual_x
                gy = (ph * ty + py) * height  # getting actual_y
                gw = (pw * np.exp(tw)) * width  # getting actual_w
                gh = (ph * np.exp(th)) * height  # getting actual_h

                xmin = np.clip(round(gx - gw / 2, 2), 0, width)
                ymin = np.clip(round(gy - gh / 2, 2), 0, height)
                xmax = np.clip(round(gx + gw / 2, 2), 0, width)
                ymax = np.clip(round(gy + gh / 2, 2), 0, height)

                class_id = np.argmax(pred_confidence_[idx, :3])

                lst = [class_id, xmin, ymin, xmax, ymax]

                for elem in lst:
                    file.write(str(elem))
                    file.write(" ")
                file.write('\n')

        
        mAP = generate_mAP(ann_confidence_[0].numpy(), pred_confidence_, thresholds)
        print(f"mAP: {mAP:.3f}")

        visualize_pred(f"test_{i}", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        # print(f"saving test_{i}...")
        cv2.waitKey(1000)