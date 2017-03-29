import cv2
import dlib
import numpy
import sys
import glob
import shutil


def read_im_and_landmarks(fname):
    
    
    return im

im = cv2.imread("Sunglasses-PNG-File.png", cv2.IMREAD_COLOR)
im2 = cv2.resize(im, (720, 360))


points  = numpy.matrix([[0, 0] for i in range(68)])
# print points

cv2.circle(im2,tuple([75,125]), 3, (0,0,255), 5)  #0
cv2.circle(im2,tuple([80,250]), 3, (0,0,255), 5)  #1
cv2.circle(im2,tuple([130,75]), 3, (0,0,255), 5)  #17
cv2.circle(im2,tuple([165,25]), 3, (0,0,255), 5)  #18
cv2.circle(im2,tuple([220,20]), 3, (0,0,255), 5)  #19
cv2.circle(im2,tuple([265,60]), 3, (0,0,255), 5)  #20
cv2.circle(im2,tuple([310,75]), 3, (0,0,255), 5)  #21
cv2.circle(im2,tuple([180,150]), 3, (0,0,255), 5) #30
cv2.circle(im2,tuple([220,125]), 3, (0,0,255), 5) #32
cv2.circle(im2,tuple([260,125]), 3, (0,0,255), 5) #38
cv2.circle(im2,tuple([290,160]), 3, (0,0,255), 5)   #39
cv2.circle(im2,tuple([265,170]), 3, (0,0,255), 5)   #40
cv2.circle(im2,tuple([230,180]), 3, (0,0,255), 5)   #41

cv2.circle(im2,tuple([390,75]), 3, (0,0,255), 5)  #22
cv2.circle(im2,tuple([430,40]), 3, (0,0,255), 5)  #23
cv2.circle(im2,tuple([480,25]), 3, (0,0,255), 5)  #24
cv2.circle(im2,tuple([535,30]), 3, (0,0,255), 5)  #25
cv2.circle(im2,tuple([580,65]), 3, (0,0,255), 5)  #26
cv2.circle(im2,tuple([634,95]), 3, (0,0,255), 5)  #16
cv2.circle(im2,tuple([650,250]), 3, (0,0,255), 5)  #15

cv2.circle(im2,tuple([420,150]), 3, (0,0,255), 5)  #42
cv2.circle(im2,tuple([450,110]), 3, (0,0,255), 5)  #43
cv2.circle(im2,tuple([490,100]), 3, (0,0,255), 5)  #44
cv2.circle(im2,tuple([520,130]), 3, (0,0,255), 5)  #45
cv2.circle(im2,tuple([470,160]), 3, (0,0,255), 5)  #47
cv2.circle(im2,tuple([500,150]), 3, (0,0,255), 5)  #46

points[0] = [75,125]
points[1] = [80,250]
points[17] = [130,75]
points[18] = [165,25]
points[19] = [220,20]
points[20] = [265,60]
points[21] = [310,75]
points[30] = [180,150]
points[32] = [220,125]
points[38] = [260,125]
points[39] = [290,160]
points[40] = [265,170]
points[41] = [230,180]

points[22] = [390,75]
points[23] = [430,40]
points[24] = [480,25]
points[25] = [535,30]
points[26] = [580,65]
points[16] = [634,95]
points[15] = [650,250]
points[42] = [420,150]
points[43] = [450,110]
points[44] = [490,100]
points[45] = [520,130]
points[47] = [470,160]
points[46] = [500,150]

numpy.save("sunglasses", points)

# print points
cv2.imshow("sunglasses",im2)
cv2.waitKey(0)