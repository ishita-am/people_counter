# import the necessary packages
from classes.centroidtracker import CentroidTracker
from classes.trackableobject import TrackableObject
# imutils - for open-cv convenience functions
# VideoStream and FPS - will help to work with web-cam and calculate estimated FPS
from imutils.video import VideoStream
from imutils.video import FPS
import winsound
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

sum = 0
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# path to caffe deploy model
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
# caffe pretrained cnn model
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
# path to input video file , if no path is provided then web-cam is used
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
# video to record here , if no path provided the video will not be recorded
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
# default value is 3, will ring alarm after 'count' number of people are in the frame
ap.add_argument("-d","--count1", type=int, default=3, help="minimum count of the people to allow in the frame")
# default value is 0.4, min probability of threshold helps to filter weak detections
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
# skip 24 frames
ap.add_argument("-s", "--skip-frames", type=int, default=24, help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# load our pre-trained MobileNet SSD used to detect object
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args ["prototxt"], args ["model"])

# if a video path was not supplied, grab a reference to the web-cam
# web-cam video stream
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
# from input file which is provided in the input
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args ["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions - we'll need to plug these into cv2.VideoWriter
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our d-lib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)  # calling the centroid-tracker class
trackers = []  # list to store d-lib correlation trackers.
trackableObjects = {}  # dict which maps an objectId to a Trackable-object

# total number of frames processed
# total number of people moved either up or down
totalFrames = 0
inFrame = 0
# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either VideoCapture or VideoStream
    frame = vs.read()
    frame = frame [1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we have reached the end of the video
    if args ["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for d-lib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    # grab the dimensions of the frame
    if W is None or H is None:
        (H, W) = frame.shape [:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args ["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args ["output"], fourcc, 30, (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args [
        "skip_frames"] == 0:  # using % operator we ensure that we'll only execute code in if-statement every N-frames
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []  # new list of trackers

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape [2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections [0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > args ["confidence"]:
                # extract the index of the class label from the
                # detections list
                idx = int(detections [0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES [idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections [0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

   

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            
            
            # check to see if the object is within the frame limits or not
            if centroid[0] > 0 and centroid[0] <W and centroid[1] >0 and centroid[1] <H:
                # if yes, count the object 
                inFrame +=1

                # check to see if number of people in the frame exceeds the limit
                if inFrame > args ["count1"]:
                    # if yes, play the alarm
                    #playsound(r"C:\Users\Admin\Desktop\ppl\Resources\alarm.mp3")
                    winsound.PlaySound(r"C:\Users\Admin\Desktop\ppl\Resources\alarm.wav", winsound.SND_ASYNC | winsound.SND_ALIAS )
                    



        # store the trackable object in our dictionary
        trackableObjects [objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid [0] - 10, centroid [1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid [0], centroid [1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Status", status),
        ("People in the Frame",inFrame)
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    inFrame=0
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()

