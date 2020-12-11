import numpy as np
import matplotlib.pyplot as plt
import cv2

orb = cv2.ORB_create(nfeatures = 20)
img_path = "dice-d4-d6-d8-d10-d12-d20/dice/valid/d20/IMG_4874.JPG"
img = cv2.imread(img_path,0)
kp, des = orb.detectAndCompute(img,None)
kp, des = orb.detectAndCompute(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()