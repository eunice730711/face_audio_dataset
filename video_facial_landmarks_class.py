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
import scenedetect

# emotion detection
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

class PySceneDetectArgs(object):
	def __init__(self, input, type='content'):
		self.input = input
		self.detection_method = type
		self.threshold = None
		self.min_percent = 95
		self.min_scene_len = 15
		self.block_size = 8
		self.fade_bias = 0
		self.downscale_factor = 1
		self.frame_skip = 2
		self.save_images = False
		self.start_time = None
		self.end_time = None
		self.duration = None
		self.quiet_mode = True
		self.stats_file = None


def cut_scene(video_path="./data/test.mp4"):
	scene_detectors = scenedetect.detectors.get_available()
	smgr_content   = scenedetect.manager.SceneManager(PySceneDetectArgs(input=video_path, type='content'),   scene_detectors)
	video_fps, frames_read, frames_processed = scenedetect.detect_scenes_file(path=video_path, scene_manager=smgr_content)
	scene_list = smgr_content.scene_list
	scene_list_msec = [(1000.0 * x) / float(video_fps) for x in scene_list]
	scene_list_tc = [scenedetect.timecodes.get_string(x) for x in scene_list_msec]
	return scene_list_tc

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
	# num_frames = int(videoclip.fps * videoclip.duration)
	# print(videoclip.fps)
	return videoclip, videoclip.fps


def face_detect(videoclip, detector, predictor):
	'''針對每個frame進行臉部偵測'''

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	frame_iter =0
	# 記錄每個frame是否有出現臉
	check_detect = {}
	left_dis = []
	right_dis = []

	# loop over the frames from the video stream
	for frame in videoclip.iter_frames(fps=18):

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
			for (x, y ) in shape:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		# 記錄每個frame是否有偵測到臉 
		if len(rects)!=0: #表示此frame有偵測到臉
			check_detect[frame_iter]=(dis_l, dis_r)
		else: # 表示此frame沒有偵測到臉
			check_detect[frame_iter]=None
		frame_iter+=1

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	return left_dis, right_dis, check_detect

def write_dataset(videoclip, left_dis, right_dis, check_detect, outfile, normalize=True):
	'''將聲音數值與畫面輸出'''

	# 取出影片聲音部份
	audioclip = videoclip.audio
	# 計算影像部份
	video_iter = 0
	# 計算聲音frame數
	frame_iter = 1
	# 訓練資料輸入值
	inseq = []
	# 訓練資料輸出直
	outseq = []
	
	# Count average distance between eyebrows and nose
	if len(left_dis)==0 and len(right_dis)==0:
		return
	left_avg = Get_Average(left_dis)
	right_avg = Get_Average(right_dis)

	# 將值normalize到-1~1之間
	if normalize==True:
		check_detect = avg_normalize(check_detect, left_avg, right_avg)

	x_seq = [] # 每次紀錄2450個點

	# record frame picture and audio
	for frame in audioclip.iter_frames(fps=44100): # normal sample numbers
		x_seq.append(frame[0]) #僅取單聲道(作為training data的input)
		if frame_iter == 2450:
			if check_detect[video_iter]!=None: # 若此frame有偵測到臉則紀錄聲音與表情
				inseq.append(x_seq) 
				frame_iter =0
				x, y = check_detect[video_iter]
				if normalize==True: # 參數決定將值normalize到-1~1之間
					outseq.append((x, y))
				else:
					outseq.append((x-left_avg, y-right_avg))# (作為training data的output)
			x_seq = []
			video_iter+=1
		# else: # 若此frame無偵測到臉則紀錄聲音 表情設定(0,0)=中性表情
		# 	inseq.append(frame[0]) #僅取單聲道(作為training data的input)
		# 	outseq.append((0, 0))
		frame_iter+=1

	# 存成npz(output filename暫定mynpz.npz)
	inseq = np.asarray(inseq)
	outseq = np.asarray(outseq)
	if len(outseq)==0:
		return
	np.savez("./output/" + outfile, x = inseq , y = outseq)

def emotion_detect(video):
	'''針對每個frame進行臉部偵測'''

	frame_iter =0
	check_detect = {}
	# hyper-parameters for bounding boxes shape
	# For Emotion Detections
	# parameters for loading data and images
	emotion_model_path = './models/emotion_model.hdf5'
	emotion_labels = get_labels('fer2013')

	# loading models
	face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
	emotion_classifier = load_model(emotion_model_path)
	# getting input model shapes for inference
	emotion_target_size = emotion_classifier.input_shape[1:3]
	# starting lists for calculating modes
	emotion_window = []
	frame_window = 10
	emotion_offsets = (20, 40)
	cap = cv2.VideoCapture(video)
	num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("frame_num: " + str(num_frames))

	# initialize check_detect
	initial = []
	for item in range(num_frames+10):
		initial.append(item)

	check_detect = dict.fromkeys(initial)
	# loop over the frames from the video stream
	while cap.isOpened(): # True:
		ret, bgr_image = cap.read()

		# grab the frame from the threaded video stream, resize it to
		# have a maximum width of 400 pixels, and convert it to
		# grayscale
		if ret is True:
			gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
			rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
		else:
			continue 
		if frame_iter == num_frames:
			break

		faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

		# 偵測到多張臉或是沒偵測到臉
		if(len(faces)!=1):
			check_detect[frame_iter] = None
			frame_iter+=1
			continue

		for face_coordinates in faces:
			x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
			gray_face = gray_image[y1:y2, x1:x2]
			try:
				gray_face = cv2.resize(gray_face, (emotion_target_size))

			except:
				continue
			gray_face = preprocess_input(gray_face, True)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)
			emotion_prediction = emotion_classifier.predict(gray_face)
			emotion_probability = np.max(emotion_prediction)
			emotion_label_arg = np.argmax(emotion_prediction)
			emotion_text = emotion_labels[emotion_label_arg]
			emotion_window.append(emotion_text)
			if len(emotion_window) > frame_window:
				emotion_window.pop(0)
			try:
				emotion_mode = mode(emotion_window)
			except:
				continue
			if emotion_text == 'angry':
				color = emotion_probability * np.asarray((255, 0, 0))
				check_detect[frame_iter]=1
			elif emotion_text == 'sad':
				color = emotion_probability * np.asarray((0, 0, 255))
				check_detect[frame_iter]=2
			elif emotion_text == 'happy':
				color = emotion_probability * np.asarray((255, 255, 0))
				check_detect[frame_iter]=3
			elif emotion_text == 'surprise':
				color = emotion_probability * np.asarray((0, 255, 255))
				check_detect[frame_iter]=4
			else:
				color = emotion_probability * np.asarray((0, 255, 0))
				check_detect[frame_iter]=5
			color = color.astype(int)
			color = color.tolist()
			draw_bounding_box(face_coordinates, rgb_image, color)
			draw_text(face_coordinates, rgb_image, emotion_mode,color, 0, -45, 1, 1)
		frame_iter+=1

		# bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		# cv2.imshow('window_frame', bgr_image)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	break
	cap.release()
	# cv2.destroyAllWindows()
	return check_detect

def write_labels(videoclip, check_detect, outfile, videofps):
	'''將聲音數值與畫面輸出'''

	# 取出影片聲音部份
	audioclip = videoclip.audio
	# 計算影像部份
	video_iter = 0
	# 計算聲音frame數
	frame_iter = 1
	# 訓練資料輸入值
	inseq = []
	# 訓練資料輸出直
	outseq = []
	x_seq = [] 

	train_len = 1000
	video_frames = len(check_detect)
	# record frame picture and audio
	for frame in audioclip.iter_frames(fps=train_len*videofps): # normal sample numbers
		x_seq.append(frame[0]) #僅取單聲道(作為training data的input)
		if frame_iter == 1000: # 每次紀錄2450個點
			if check_detect[video_iter]!=None: # 若此frame有偵測到臉則紀錄聲音與表情
				inseq.append(x_seq) 
				label = check_detect[video_iter]
				outseq.append((x_seq, label))
			x_seq = []
			video_iter +=1
			frame_iter =0
		frame_iter+=1
		if(video_iter>=video_frames-1):
			break
	# 存成npz(output filename暫定mynpz.npz)
	inseq = np.asarray(inseq)
	outseq = np.asarray(outseq)
	if len(outseq)==0:
		return
	np.savez("./output/" + outfile, x = inseq , y = outseq)

	

if(__name__=='__main__'):

	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--video", required=True,
		help="path to video file")
	ap.add_argument("-m", "--mode", required=True,
		help="data collection mode: regression or classify")
	args = vars(ap.parse_args())

	# 讀進Vedio
	myclip, fps =read_vedio(args["video"])

	# 自動切 & 人臉位置偵測
	if args["mode"] == 'regression':
		cut_list = cut_scene(args["video"])
		print("[INFO] loading facial landmark predictor...")
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

		idx = 0
		cut_list = ['00:00:00.000'] + cut_list
		t_start = ""
		t_end = ""
		print(cut_list)
		file_idx = 0
		for item in cut_list:
			if idx%2!=0:
				t_end = item
				print(t_start, t_end)
				left_dis, right_dis, check_detect = face_detect(myclip.subclip(t_start, t_end), detector, predictor)
				filename = args["video"][7:-4]+str(file_idx)
				file_idx+=1
				write_dataset(myclip.subclip(t_start, t_end), left_dis, right_dis, check_detect, filename)
			else:
				t_start = item
			idx+=1

	elif args["mode"] == 'classify':  # 情緒分類
		check_labels = {}
		outfile = args["video"][7:-4]
		check_labels = emotion_detect(args["video"])
		write_labels(myclip, check_labels, outfile, fps)
	del myclip







