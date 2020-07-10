#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as standard_transforms

from flask import Flask, render_template, Response
from camera import VideoCamera

import numpy
import time
import cv2
import csv

from PIL import Image

#EfficientNet-b7
from models.SCC_Model.EfficientNet_SFCN import EfficientNet_SFCN as net
from models.CC import CrowdCounter
CCN = CrowdCounter([0],'EfficientNet_SFCN')
CCN.load_state_dict(torch.load('models/all_ep_171_mae_11.6_mse_19.7.pth'))  # EfficientNet-b7 Modified - 200 Epoch (EfficientNet_SFCN)

CCN.CCN.res._blocks = CCN.CCN.res._blocks[0:18]

# FPNCC
# from models.SCC_Model.Res101_FPN import Res101_FPN as net
# from models.CC import CrowdCounter
# CCN = CrowdCounter([0],'Res101_FPN')
# CCN.load_state_dict(torch.load('models/fpncc_shhb.pth'))  # FPNCC - 200 Epoch (Res101_FPN)

print("Model successfully loaded")

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# FPNCC
# def gen(camera):
#     frames_to_read = 9    # max frames_to_read = 9
#     i = 0
#     images = []

#     while True:
#         if i%frames_to_read==0:
#             start = time.time()
#             images = [] 
        
#         frame, image = camera.get_frame()   # numpy array

#         img_transform = standard_transforms.Compose([
#             standard_transforms.Resize((240,320)),      # Resize PIL
#             standard_transforms.ToTensor(),             # convert PIL -> tensor
#             standard_transforms.Normalize(*mean_std)    # normalize numpy array
#         ])

#         image = Image.fromarray(image.astype('uint8'), 'RGB')   #convert numpy array ke PIL

#         image = img_transform(image).cuda()

#         images.append(image) #masukkin image ke arraynya images
        
#         i += 1

#         if i%frames_to_read==0:
#             images = torch.stack(images, axis=0)
#             CCN.CCN.eval()

#             with torch.no_grad():
#                 density_map = CCN.CCN(images)
#                 count = density_map.sum(axis=(2,3)).mean()/100.        
            
#             print(count)

#             end = time.time()
#             seconds = end - start
#             fps = frames_to_read/seconds
#             print(fps)

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# EfficientNet b7
def gen(camera):
    frames_to_read = 13 # max frames_to_read = 13
    i = 0
    images = []

    while True:
        if i%frames_to_read==0:
            start = time.time()
            images = [] 
        
        frame, image = camera.get_frame()   # numpy array

        img_transform = standard_transforms.Compose([
            standard_transforms.Resize((240,320)),      # Resize PIL
            standard_transforms.ToTensor(),             # convert PIL -> tensor
            standard_transforms.Normalize(*mean_std)    # normalize numpy array
        ])

        image = Image.fromarray(image.astype('uint8'), 'RGB')   #convert numpy array ke PIL

        image = img_transform(image).cuda()

        images.append(image) #masukkin image ke arraynya images
        
        i += 1

        # import pdb; pdb.set_trace()

        if i%frames_to_read==0:
            images = torch.stack(images, axis=0)
            CCN.CCN.eval()

            with torch.no_grad():
                density_map = CCN.CCN(images)
                count = density_map.sum(axis=(2,3)).mean()/100.        
            
            print(count)

            end = time.time()
            seconds = end - start
            fps = frames_to_read/seconds
            print(fps)

            # Write Crowd Counting Result to CSV File
            csvFile = "crowd_counting_result.csv"
            with open(csvFile, 'a') as fp:
                wr = csv.writer(fp)
                wr.writerow([count]) 

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
	return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)