# -*- coding: utf-8 -*-
# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import moviepy
import math
from moviepy.editor import *
 
def read_vedio(video_file="./data/test.mp4"):
	# loading the video 
	print("[INFO] loading the vedio...")
	videoclip = VideoFileClip(video_file)
	return videoclip

def face_detect(videoclip, dlib_detector="shape_predictor_68_face_landmarks.dat"):

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(dlib_detector)
	posx =0
	posy =0
	frame_iter =0
	face_pos = {}
	frame_rec = {}

	# loop over the frames from the video stream
	for frame in myclip.iter_frames(fps=18):

		# grab the frame from the threaded video stream, resize it to
		# have a maximum width of 400 pixels, and convert it to
		# grayscale
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)
		frame_iter+=1

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			# 顎1 ~ 17; 左眉18 ~ 22; 右眉23 ~ 27; 鼻子28 ~ 31,32 ~ 36; 左眼37 ~ 42; 右眼43 ~ 48; 口49 ~ 68
			tmp_pos = []
			for (x, y) in shape:
				posx = x-shape[28][0]
				posy = y-shape[28][0]
				tmp_pos.append([posx,posy])
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		# Record face detection 
		if len(rects)!=0:
			face_pos[frame_iter]=tmp_pos
			frame_rec[frame_iter]=frame
		else:
			face_pos[frame_iter]=[]
			frame_rec[frame_iter]=None

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	print(frame_iter)
	return face_pos, frame_rec


def write_dataset(videoclip,face_pos, frame_rec):
	audioclip = videoclip.audio
	frame_iter = 0
	frame_count = 0
	s = 0 # start time
	record = []
	fra_rec = []
	for frame in audioclip.iter_frames(fps=18):
		frame_iter+=1
		frame_count+=1
		# Record at every second
		if frame_count==18:
			frame_count=0
			# check face appear and record at this time step
			if len(face_pos[frame_iter])!=0:
				record = face_pos[frame_iter]
				fra_rec = frame_rec[frame_iter]

			# output data to files
			if len(record)!=0:
				audioclip.subclip(s,s+1).write_audiofile("./output/aaa"+str(s)+".wav")
				cv2.imwrite("./output/aaa"+str(s)+".png",fra_rec)
				record=[]
			s+=1
		elif len(face_pos[frame_iter])!=0:
			record = face_pos[frame_iter]
			fra_rec = frame_rec[frame_iter]



if(__name__=='__main__'):

	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--video", required=True,
		help="path to video file")

	args = vars(ap.parse_args())

	myclip=read_vedio(args["video"])
	face_pos, frame_rec = face_detect(myclip)
	write_dataset(myclip,face_pos,frame_rec)