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

def Get_Average(list):
	sum = 0
	for item in list:
		sum += item
	return sum/len(list)


def avg_normalize(dic,lavg,ravg):
	eyes = []
	max_ = 0
	min_ = 1e10
	# Find max_ and min_ in detect_check
	for key, value in dic.items():
		if value!=None:
			eyes.append((key,(value[0]-lavg,value[1]-ravg)))
			if value[0]-lavg < min_:
				min_ = value[0]-lavg
			if value[0]-lavg > max_:
				max_ = value[0]-lavg
			if value[1]-ravg < min_:
				min_ = value[1]-ravg
			if value[1]-ravg > max_:
				max_ = value[1]-ravg
		else:
			eyes.append((key,None))

	# Calculate normalize values
	new_dic = {}
	avg = (min_ + max_)/2
	div = max_/2 - min_/2
	# print(avg)
	# print(div)

	# Get a new normalize dic
	for key, value in eyes:
		if value==None:
			new_dic[key]=None
		else:
			newx = (value[0]-avg)/div
			newy = (value[1]-avg)/div
			new_dic[key]=(newx,newy)
	return new_dic


def read_vedio(video_file="./data/test.mp4"):
	# loading the video 
	print("[INFO] loading the vedio...")
	videoclip = VideoFileClip(video_file)
	return videoclip


def face_detect(videoclip, dlib_detector="shape_predictor_68_face_landmarks.dat"):
	'''針對每個frame進行臉部偵測'''

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(dlib_detector)
	frame_iter =0
	# 記錄每個frame是否有出現臉
	check_detect = {}
	left_dis = []
	right_dis = []

	# loop over the frames from the video stream
	for frame in myclip.iter_frames(fps=18):

		# grab the frame from the threaded video stream, resize it to
		# have a maximum width of 400 pixels, and convert it to
		# grayscale
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)
		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			# 顎1 ~ 17; 左眉18 ~ 22; 右眉23 ~ 27; 鼻子28 ~ 31,32 ~ 36; 左眼37 ~ 42; 右眼43 ~ 48; 口49 ~ 68
			# 左眉中心20 右眉中心25 鼻子中心31
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# 左眉中心位置
			tmp_left = (shape[19][0], shape[19][1]) 
			# 右眉中心位置
			tmp_right = (shape[24][0], shape[24][1])
			# 鼻子中心位置
			tmp_nose = (shape[30][0], shape[30][1])
			# 計算左眉中心與右眉中心到鼻子中心的直線距離
			dis_l = ((tmp_left[0] - tmp_nose[0])**2 + (tmp_left[1] - tmp_nose[1])**2)**0.5
			dis_r = ((tmp_right[0] - tmp_nose[0])**2 + (tmp_right[1] - tmp_nose[1])**2)**0.5

			left_dis.append(((tmp_left[0] - tmp_nose[0])**2 + (tmp_left[1] - tmp_nose[1])**2)**0.5)
			right_dis.append(((tmp_right[0] - tmp_nose[0])**2 + (tmp_right[1] - tmp_nose[1])**2)**0.5)

			# 測試：印在畫面上看看眉毛與鼻子位置是否符合
			# test_list =[]
			# test_list = [tmp_left, tmp_right, tmp_nose]
			# for (x, y ) in test_list:
			# 	cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		# 記錄每個frame是否有偵測到臉 
		if len(rects)!=0: #表示此frame有偵測到臉
			check_detect[frame_iter]=(dis_l, dis_r)
		else: # 表示此frame沒有偵測到臉
			check_detect[frame_iter]=None
		frame_iter+=1

		# show the frame
		# cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	# cv2.destroyAllWindows()
	return left_dis, right_dis, check_detect


def write_dataset(videoclip, left_dis, right_dis, check_detect, normalize=True):
	'''將聲音數值與畫面輸出'''

	# 取出影片聲音部份
	audioclip = videoclip.audio
	# 計算frame數
	frame_iter = 0
	# 訓練資料輸入值
	inseq = []
	# 訓練資料輸出直
	outseq = []
	# Left = []
	# Right = []
	
	# Count average distance between eyebrows and nose
	left_avg = Get_Average(left_dis)
	right_avg = Get_Average(right_dis)

	# 將值normalize到-1~1之間
	if normalize==True:
		check_detect = avg_normalize(check_detect, left_avg, right_avg)

	# record frame picture and audio
	for frame in audioclip.iter_frames(fps=18):
		if check_detect[frame_iter]!=None: # 若此frame有偵測到臉則紀錄聲音與表情
			inseq.append(frame[0]) #僅取單聲道(作為training data的input)
			x, y = check_detect[frame_iter]
			if normalize==True: # 參數決定將值normalize到-1~1之間
				outseq.append((x, y))
			else:
				outseq.append((x-left_avg, y-right_avg))# (作為training data的output)

		else: # 若此frame無偵測到臉則紀錄聲音 表情設定(0,0)=中性表情
			inseq.append(frame[0]) #僅取單聲道(作為training data的input)
			outseq.append((0, 0))

		frame_iter+=1
	# # 取得臉部表情標記
	# face = []
	# for l,r in zip(left_dis,right_dis):
	# 	face.append((l-left_avg,r-right_avg))

	# # Create distance data
	# for item in left_dis :
	# 	Left.append(item - left_avg)
	# for item in right_dis :
	# 	Right.append(item - right_avg)


	# 存成npz(output filename暫定mynpz.npz)
	inseq = np.asarray(inseq)
	outseq = np.asarray(outseq)
	np.savez("./output/mynpz.npz", x = inseq , y = outseq)
	
		
if(__name__=='__main__'):

	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--video", required=True,
		help="path to video file")

	args = vars(ap.parse_args())

	myclip=read_vedio(args["video"])
	left_dis, right_dis, check_detect = face_detect(myclip)
	write_dataset(myclip, left_dis, right_dis, check_detect)

