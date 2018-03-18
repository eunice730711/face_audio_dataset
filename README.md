# face_audio_dataset
https://hackmd.io/CwVgHAnAbAJsUFoDsSBGAzBwDGMCGCkUECUIAjAExRQAMw5ApgMypA==?view

Latest:
video_facial_landmards_class.py
usage: 
1) python video_facial_landmarks_class.py -f ./data/test.mp4 -m classify
2) python video_facial_landmarks_class.py -f ./data/test.mp4 -m regression (same as autocut version) 

目前檔案說明：

old_version: video_facial_landmarks.py
每秒產出對應的照片與wav檔

new_version: video_facial_landmards_new.py
每秒製作18個frame，聲音(單聲道)對應影片中眉毛的位置改變(取左眉與右眉到鼻子的直線距離）
輸出在”./output/mynpz.npz”

new_version1: video_facial_landmards_class.py
兩種不同資料收集： regression mode:眉毛偵測  classify mode: 情緒分類

read_npz.py: npz讀檔測試

Usage on test files：
python video_facial_landmarks_new.py -f ./data/test.mp4
影片中紅色三個位置點即為label位置

References:
https://github.com/petercunha/Emotion