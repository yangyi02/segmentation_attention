#!/usr/bin/python
# 

import cv2
import sys
import os
import glob

def preprocessing(videos_list):
    with open(videos_list, "r") as f:
        files = f.read().splitlines()

    for v in files:
        print v

        video_dir = os.path.splitext(v)[0]
        imgs_dir = video_dir + "/imgs"
        video_name = os.path.split(video_dir)[1]

        os.system("mkdir -p " + imgs_dir)

        cap = cv2.VideoCapture(v)
        frames_n = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        # skip already extracted 
        if len(glob.glob("%s/*.jpg" % imgs_dir)) != frames_n:
            cnt = 1
            while True:
                success, frame = cap.read()
                if not success:
                    break
                cv2.imwrite("%s/%06d.jpg" % (imgs_dir,cnt), frame)
                cnt += 1

        cap.release()

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "usage: ./extract_frames.py <videos-list>"
        sys.exit()    

    preprocessing(sys.argv[1])

    print "done."
    
