# -*- coding: utf-8 -*-
import numpy as np
npzfile = np.load("./output/test23.npz")
count=0
print("Length of npz file:")
print(len(npzfile['x']))
print("前1000筆數值:")
for x, y in zip(npzfile['x'],npzfile['y']):
	# print((x,y))
	print(len(x),len(y))
	if count==1000:
		break
	count+=1