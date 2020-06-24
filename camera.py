import cv2
import time

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
        fps = self.video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0} fps".format(fps))

        # # Number of frames to capture
        # num_frames = 120;
        
        # print("Capturing {0} frames".format(num_frames))

        # # Start time
        # start = time.time()
        
        # # End time
        # end = time.time()

        # # Time elapsed
        # seconds = end - start
        # print("Time taken : {0} seconds".format(seconds))

        # # Calculate frames per second
        # fps  = num_frames / seconds;
        # print("Estimated frames per second : {0}".format(fps))

        # frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        # print("Estimated frame rate : {0}".format(frame_count))

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()