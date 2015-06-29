# -*- coding: utf-8 -*-
import cv2
import os
import sys

temp_name_list = os.listdir(sys.argv[1])
sizex = sys.argv[2]
sizey = sys.argv[3]


base_temp = []
for name in temp_name_list:
  img = cv2.resize(cv2.imread(sys.argv[1] + "/" + name),(int(sizex),int(sizey)))
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img2 = img_gray.copy()
  for i in range(0,5):
    img2 = cv2.medianBlur(img2,3)
    cv2.equalizeHist(img2,img2)#平坦化
  base_temp.append(img2)

i=0
for image in base_temp:
  cv2.imwrite("templates/%d.png"%i,image)
  i+=1