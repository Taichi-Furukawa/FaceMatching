# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
img = cv2.imread(sys.argv[1])#元画像読み込み
img = cv2.resize(img,(512,512))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#グレイ画像作成
img_blur = img_gray.copy()#ブラー用コピー
for i in range(0,5):
	img_blur = cv2.medianBlur(img_blur,3)#フィルタでぼかす平滑化	
	cv2.equalizeHist(img_blur,img_blur)#ヒストグラム平坦化
cv2.imwrite("assets/matching.png",img_blur)#保存

#cv2.equalizeHist(img_gray,img_gray)

img_sobel = cv2.Sobel(img_gray,3,1,0)#ソーベル微分画像
#img_sobel_x = cv2.Sobel(img_gray,3,1,0)#ソーベル微分画像
#a = np.power(img_sobel_x,2)+np.power(img_sobel_y,2)
#img_sobel = np.power(a,0.5)

cv2.imwrite("assets/sobelBinary.png",img_sobel)#保存

#img_sobel = cv2.cvtColor(img_sobel, cv2.COLOR_BGR2GRAY)#グレイ画像作成
mask = img_sobel > 127#閾値５０で白黒に
img_sobel_bi = np.zeros((img_sobel.shape[0],img_sobel.shape[1]),np.uint8)
img_sobel_bi[mask] = 255
cv2.imwrite("assets/sobelBinary.png",img_sobel_bi)#保存


