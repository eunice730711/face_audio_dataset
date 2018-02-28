# -*- coding: utf-8 -*-
import numpy as np
npzfile = np.load("./output/mynpz.npz")
count=0
print("Length of npz file:")
print(len(npzfile['x']))
print("前100筆數值:")
for x, y in zip(npzfile['x'],npzfile['y']):
	print((x,y))
	if count==100:
		break
	count+=1