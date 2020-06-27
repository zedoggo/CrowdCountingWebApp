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

# from models.SCC_Model.EfficientNet_SFCN import EfficientNet_SFCN as net
from models.SCC_Model.Res101_FPN import Res101_FPN as net
from models.CC import CrowdCounter

CCN = CrowdCounter([0],'Res101_FPN')
# density_map = CCN(img)
# CCN.load_state_dict(torch.load('models/all_ep_171_mae_11.6_mse_19.7.pth'))  # EfficientNet-b7 Modified - 200 Epoch (EfficientNet_SFCN)
CCN.load_state_dict(torch.load('models/fpncc_shhb.pth'))  # FPNCC - 200 Epoch (Res101_FPN)
print("Model successfully loaded")

from flask import Flask, render_template, Response
from camera import VideoCamera

import time
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
    	# start time
        start = time.time()
        frame = camera.get_frame()
        end = time.time()
        seconds = end - start
        fps = 1./seconds
        print(fps)
        #####
        # end time
        # interval
        # fps = 1/interval
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
	return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)