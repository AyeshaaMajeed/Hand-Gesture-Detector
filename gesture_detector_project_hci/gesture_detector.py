#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import os
from utils import clPreProcessing
from utils import clTraningSetManager
from utils import ContourDetector
from utils import clAutoCalibrate
from utils import clHogDetector
import math

# main loop
if __name__ == "__main__":

    # process cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=str, required=True, metavar='labelfile', help='label file, like ./label.txt')
    parser.add_argument('-t', type=str, required=False, metavar='trainedfile',
                        help='file which holds the trained values like, ./data.xml')
    parser.add_argument('-cmd', type=str, metavar='commands', dest='choises',
                        choices=['retrain', 'savedetections', 'creatlabelfile', 'run'], default='run',
                        help='commands available for processing: [retrain, savedetections, creatlabelfile, run], default=run')
    parser.add_argument('-d', type=int, metavar='detectionswindow', default=0, choices=[0,1],
                        help='show/ hide detection window [0,1], default=0')
    parser.add_argument('-c', type=int, metavar='cameraid', default=0, choices=[0,1],
                        help='use different camera ID, default=0')
    parser.add_argument('-s', type=int, metavar='samplesize', default=96, choices=[32,64,96,192],
                        help='select the training sample size, default=96')
    parser.add_argument('-dir', type=str, required=False, metavar='traindirectory', help='directory with ordered pictures per detection classes')
    parser.add_argument('-dd', type=str, required=False, metavar='detectiondir', help='directory to save the detections or image patches for training set creation')
    parser.add_argument('-ani', type=int, required=False, metavar='annotatedimages', default=0, choices=[0,1],
                        help='save annotated images, [0,1] default=0')
    parser.add_argument('-sf', type=int, required=False, metavar='saveframe', default=0, choices=[0,1],
                        help='save original camera frames, [0,1] default=0')
    parser.add_argument('-sp', type=int, required=False, metavar='savepatches', default=0, choices=[0,1],
                        help='save image detections (patches), [0,1] default=0')

    parser.add_argument('-cal', type=int, required=False, metavar='calibration', default=0, choices=[0,1],
                        help='reclalibrate skin color detection, [0,1] default=0')

    args = parser.parse_args()
    cmd = args.choises

    # define image dimensions
    IMG_WIDTH = 320
    IMG_HEIGHT = 240
    CAMID = args.c
    SAMPLESIZE = args.s

    labelfile = args.l
    trainedfile = args.t
    traindir = args.dir
    detectionsdir = args.dd

    # create named window, set position
    cv2.namedWindow('img', 2)
    cv2.moveWindow('img', 0, 0)

    # tmp video path file based:
    current_dir = os.getcwd()
    video_path = os.path.join(current_dir, "video.mp4")
    #cam0 = cv2.VideoCapture(video_path)

    # original camera based input:
    # create cam instance
    cam0 = cv2.VideoCapture(CAMID)

    # resize, to spare CPU load
    cam0.set(3, IMG_WIDTH)
    cam0.set(4, IMG_HEIGHT)

    # create empty images
    img0 = np.zeros((IMG_WIDTH,IMG_HEIGHT,3),dtype=np.byte)
    imgd = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)

    # pre-processing
    objPP = clPreProcessing(img0, False, 150, 66, 66)

    # contour detector
    cd = ContourDetector()

    # object for data set handling
    tsm = clTraningSetManager()

    # skin color autocalibration
    ac = clAutoCalibrate()
            


    if args.cal == 1:
        #load calibration values from a file
        val = tsm.LoadLabelsFile(labelfile,True)
        objPP.SetColorFilteringThresholds(int(val[0]), int(val[1]), int(val[2]))

    if cmd == 'run':
        if labelfile is None:
            print ("Labels file is missing, use -h for available arguments")
            os._exit(0)

        # load labels file
        lf = tsm.LoadLabelsFile(labelfile)

        if trainedfile is None:
            print ("Training file is missing, use -h for available arguments")
            os._exit(0)
        #load trained file
        det = clHogDetector(SAMPLESIZE,trainedfile)
        det.UpdateLabelNames(lf)

    elif cmd =='retrain':

        if labelfile is None:
            print("Labels file is missing, use -h for available arguments")
            os._exit(0)

        if trainedfile is None:
            print ("Trained file is missing, use -h for available arguments")
            os._exit(0)

        det = clHogDetector(SAMPLESIZE)

        # load labels file
        lf = tsm.LoadLabelsFile(labelfile)

        # load training set
        for i in lf:
            uid = i[0]
            label = i[1]
            path = i[2]
            det.AddToTrainingSet(path, int(uid), label)

        # update label names
        det.UpdateLabelNames(lf)

        # train SVM
        det.TrainSVMWithHOG(SAMPLESIZE)

        # save trained file
        det.SaveTrainingData(trainedfile)

    elif cmd == 'creatlabelfile':
        if traindir is None:
            print("Argument with training directory is missing, use -h for available arguments")
            os._exit(0)
        else:
            tsm.SaveLabelsFile(traindir, labelfile)
            print ("Labelfile " + labelfile + " created, exiting.")
            os._exit(0)

    elif cmd == 'savedetections':
        # show helper windows and save detections to a folder
        if detectionsdir is None:
            print("Argument with detections directory is missing, use -h for available arguments")
            os._exit(0)
        else:
            print("Detections will be save in: " + detectionsdir)
            args.d = 1
    
    angle = 0 # angle of iamge rotation

    while (True):

        centers_arr = []

        _, img0 = cam0.read()

        # test cam instances
        if (cam0):

            img = objPP.CombineDetections(img0)
            img = objPP.processFilter(img)
            aa = cd.CotourFilter(img,500.0)

            rois = cd.GetRoiForDetections(img0,aa,0)

            imgd = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3],dtype=np.uint8)
            imgd = cd.ShowRoisOnImage(imgd,rois)

            val = det.ClassifyRoi(rois, SAMPLESIZE)
            if args.cal == 0:
                img, centers_arr = cd.DrawDetections(img0, aa, 0, True, True, val,False)
            else:
                img = ac.RunCalibration(img0)

            if detectionsdir is not None:
                if args.sp == 1:
                    cd.SaveImages(img0, rois,detectionsdir)

                if args.ani == 1:
                    cd.SaveImages(img, [], detectionsdir,prefix="ani_")

                if args.sf == 1:
                    cd.SaveImages(img0, [], detectionsdir,prefix="sf_")

            cv2.imshow('img', img)

            if args.d == 1:
                cv2.imshow('imgd', imgd)


        k = cv2.waitKey(1) & 0xFF

        # quit on keypress
        if k == ord('q'):
          break

        # calibrate
        if k == ord('c'):
            if args.cal != 0:
                # save calibration values
                val = ac.ProvideClaibParams()
                objPP.SetColorFilteringThresholds(int(val[0]), int(val[1]), int(val[2]))

                tsm.SaveCalibration(labelfile, val)

                #exit
                break
        
        # display an image image.png on right hand side
        imgg = cv2.imread('image.png')
        cv2.imshow('imgg', imgg)

        # get the position of 1st element of centers_arr and rotate the imgg based on that
        if len(centers_arr) > 0:
            center = centers_arr[0]
            # angle = math.degrees(math.atan2(center[1], center[0]))
            # interpolate between old and new angle for smooth rotation
            angle = 0.7 * math.degrees(math.atan2(center[1], center[0])) + 0.5 * angle
            imgg = cv2.imread('image.png')
            rows, cols, _ = imgg.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            imgg = cv2.warpAffine(imgg, M, (cols, rows))
            cv2.imshow('imgg', imgg)


    # release cam
    cam0.release()
    cv2.destroyAllWindows()

