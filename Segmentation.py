import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import sys

img = cv.imread("d4_angle_color060.jpg")
if img is None:
    sys.exit("No such image")
ddepth = cv.CV_64F

# Attempt at using Laplacian Edge Detection
def laplaceEdge(imgIn):
    gray = cv.cvtColor(imgIn, cv.COLOR_BGR2GRAY)
    src = cv.GaussianBlur(gray, (3, 3), 0)
    med = cv.medianBlur(src, 5)
    box = cv.boxFilter(med, -1, (3, 3))
    filtered_image = cv.Laplacian(box, ksize=3, ddepth=ddepth)
    abs_grad = cv.convertScaleAbs(filtered_image)
    cv.imshow('Laplace edges', abs_grad)
    cv.waitKey(0)
    return abs_grad

# Attempt at using Sobel Edge Detection
def sobelEdge(imgIn):
    gray = cv.cvtColor(imgIn, cv.COLOR_BGR2GRAY)
    src = cv.GaussianBlur(gray, (3, 3), 0)
    med = cv.medianBlur(src, 5)
    box = cv.boxFilter(med, -1, (3, 3))
    grad_x = cv.Sobel(box, ddepth, 1, 0, ksize=3)
    grad_y = cv.Sobel(box, ddepth, 0, 1, ksize=3)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv.imshow('Sobel edges', grad)
    cv.waitKey(0)
    return grad

def findLargestArea(contourSet):
    contidx = 0
    greatestArea = 0
    for cont in range(0, len(contourSet)):
        topEdge = 100000
        botEdge = 0
        leftEdge = 100000
        rightEdge = 0
        for edge in contourSet[cont]:
            if edge[0][0] > botEdge:
                botEdge = edge[0][0]
            if edge[0][0] < topEdge:
                topEdge = edge[0][0]
            if edge[0][1] > rightEdge:
                rightEdge = edge[0][1]
            if edge[0][1] < leftEdge:
                leftEdge = edge[0][1]
        length = abs(botEdge - topEdge)
        width = abs(rightEdge - leftEdge)
        area = length * width
        if area > greatestArea:
            greatestArea = area
            contidx = cont
    return contidx

def cannyEdges(imgIn):
    # gray = cv.cvtColor(imgIn, cv.COLOR_BGR2GRAY)
    gaussian = cv.GaussianBlur(imgIn, (3, 3), 3)
    med = cv.medianBlur(gaussian, 5)
    box = cv.boxFilter(med, -1, (3, 3))
    # Use Canny Edge ops to find edges
    # ret, thresh = cv.threshold(gray,200, 255, cv.THRESH_TOZERO_INV)
    edged = cv.Canny(box, 30, 100)
    cv.imshow('canny edges', edged)
    cv.waitKey(0)
    return edged

#Thresholding attempt
def threshold(imgIn):
    gray = cv.cvtColor(imgIn, cv.COLOR_BGR2GRAY)
    mat, thresh = cv.threshold(gray, 150, 255, cv.THRESH_TOZERO_INV)
    cv.imshow('threshold', thresh)
    cv.waitKey(0)
    return thresh

def segmentation(imgIn):
    #Use Canny Edges to find contours
    # Also try RETR_LIST, RETR_CCOMP, or RETR_TREE a try, RETR_EXTERNAL
    # If we need it, try CHAIN_APPROX_NONE instead of CHAIN_APPROX_SIMPLE
    contours, hierarchy=cv.findContours(imgIn,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    cv.imshow('edges after contouring', imgIn)
    cv.waitKey(0)
    return contours

# lapImg = laplaceEdge(img)
# sobel = sobelEdge(img)
# canny = cannyEdges(img)
threshImg = threshold(img)
contours = segmentation(threshImg)
# Print the number of contours
# print(contours)
print('Numbers of contours found=' + str(len(contours)))

die = findLargestArea(contours)
print("Largest Area: " + str(die))

# Draw the contours on the image
cv.drawContours(img,contours,-1,(0,255,0),3)
cv.imshow('contours',img)
cv.waitKey(0)
cv.destroyAllWindows()
