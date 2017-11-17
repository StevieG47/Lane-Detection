#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 06:24:33 2017

@author: steve
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os



# Get all images
test_images = [mpimg.imread('test_video/videoFrames/' + i) for i in os.listdir('test_video/videoFrames/')]
#test_image_names = ['test_video/videoFrames/'+i for i in os.listdir('test_video/videoFrames/')]


#----------------GET IMAGE-----------------
im = test_images[0]
imshape = im.shape
#plt.figure(1)
#plt.imshow(im)
#plt.title(test_image_names[0])

# -------CREATE VIDEO WRITER------------------
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoOut = cv2.VideoWriter('output.avi',fourcc, 20.0, (im.shape[1],im.shape[0]))

for frameNum in range(1,len(test_images)-1):
    #print('Frame Number: ',frameNum)
    
    im = mpimg.imread('test_video/videoFrames/'+str(frameNum)+'.jpg')
    imshape = im.shape
      
        
    # -------------GREYSCALE IMAGE---------------
    # Grayscale one color channel
    # specify cmap = 'gray' in plt.imshow
    grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #plt.figure(2)
    #plt.imshow(grayIm,cmap='gray')
    #plt.title('Greyscaled image')
    
    
    #------------GAUSSIAN SMOOTHING-----------------
    # Use low pass filter to remove noise. Will remove high freq stuff like noise and edges
    # kernel_size specifies width/height of kernel, should be positive and odd
    # Also specify stand dev in X and Y direction, give zero to calculate from kernel size
    # Can also use average, median, and bilarteral blurring techniques
    kernel_size = 9; # bigger kernel = more smoothing
    smoothedIm = cv2.GaussianBlur(grayIm, (kernel_size, kernel_size), 0)
    #plt.figure(3)
    #plt.imshow(smoothedIm,cmap='gray')
    #plt.title('Smoothed image')
    
    
    #-------------EDGE DETECTION---------------------
    # finds gradient in x,y direction, gradient direction is perpendicular to edges
    # checks pixels in gradient directions to see if they are local maximums, meaning on an edge
    # hysteresis thresholding has min and max value, edges with gradient intensity big enough are edges
    # edges that lie in bewteen are check to see if they connect to edges with intensity greater than max value, then it is considered edge
    # also assumes edges are long lines (not small pixels regions)
    minVal = 60
    maxVal = 150
    edgesIm = cv2.Canny(smoothedIm, minVal, maxVal)
    #plt.figure(4)
    #implot = plt.imshow(edgesIm,cmap='gray')
    
    #plt.scatter([0],[imshape[0]])
    #plt.scatter([465],[320])
    #plt.scatter([475],[320])
    #plt.scatter([imshape[1]],[imshape[0]])
    
    #plt.title('Edge Detection')
    
    
    #-------------------------CREATE MASK--------------------------------
    # Create mask to only keep area defined by four coners
    # Black out every area outside area
    vertices = np.array([[(0,imshape[0]),(465, 320), (475, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    
    # defining a blank mask to start with, 0s with same shape of edgesIm
    mask = np.zeros_like(edgesIm)   
              
    # fill pixels inside the polygon defined by vertices"with the fill color  
    color = 255
    cv2.fillPoly(mask, vertices, color)
    
    # show mask
    #plt.figure(5)
    #plt.imshow(mask,cmap='gray')
    #plt.title('Mask')
        
    #----------------------APPLY MASK TO IMAGE-------------------------------
    # create image only where mask and edge Detection image are the same
    maskedIm = cv2.bitwise_and(edgesIm, mask)
        
    # Plot output of mask
    #plt.figure(6)
    #plt.imshow(maskedIm,cmap='gray')
    #plt.title('Masked Image')
    
    # Plot masked edges image
    maskedIm3Channel = cv2.cvtColor(maskedIm, cv2.COLOR_GRAY2BGR)
    
    
    #-----------------------HOUGH LINES------------------------------------
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 45     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 100    # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(maskedIm, rho, theta, threshold, np.array([]), 
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
    # Draw all lines onto image
    allLines = np.zeros_like(maskedIm)
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(allLines,(x1,y1),(x2,y2),(255,255,0),2) # plot line
    
    # Plot all lines found
    #plt.figure(7)
    #plt.imshow(allLines,cmap='gray')
    #plt.title('All Hough Lines Found')
       
    #-----------------------Separate Lines Intro Positive/Negative Slope--------------------------
    # Separate line segments by their slope to decide left line vs. the right line
    slopePositiveLines = [] # x1 y1 x2 y2 slope
    slopeNegativeLines = []
    yValues = []
    
    # Loop through all lines
    addedPos = False
    addedNeg = False
    for currentLine in lines:   
        # Get points of current Line
        for x1,y1,x2,y2 in currentLine:
            lineLength = ((x2-x1)**2 + (y2-y1)**2)**.5 # get line length
            if lineLength > 30: # if line is long enough
                slope = (y2-y1)/(x2-x1) # get slope
                if slope > 0: 
                    slopeNegativeLines.append([x1,y1,x2,y2,-slope]) # add positive slope line
                    yValues.append(y1)
                    yValues.append(y2)
                    addedPos = True # note that we added a positive slope line
                if slope < 0:
                    slopePositiveLines.append([x1,y1,x2,y2,-slope]) # add negative slope line
                    yValues.append(y1)
                    yValues.append(y2)
                    addedNeg = True # note that we added a negative slope line
           
            
    # If we didn't get any positive lines, go though again and just add any positive slope lines         
    if not addedPos:
        for currentLine in lines:
            for x1,y1,x2,y2 in currentLine:
                slope = (y2-y1)/(x2-x1)
                if slope > 0:
                    slopeNegativeLines.append([x1,y1,x2,y2,-slope])
                    yValues.append(y1)
                    yValues.append(y2)
    
    # If we didn't get any negative lines, go through again and just add any negative slope lines
    if not addedNeg:
        for currentLine in lines:
            for x1,y1,x2,y2 in currentLine:
                slope = (y2-y1)/(x2-x1)
                if slope < 0:
                    slopePositiveLines.append([x1,y1,x2,y2,-slope])           
                    yValues.append(y1)
                    yValues.append(y2)
                   
    
    if not addedPos or not addedNeg:
        print('Not enough lines found')
    
    
    #------------------------Get Positive/Negative Slope Averages-----------------------------------
    # Average position of lines and extrapolate to the top and bottom of the lane.
    positiveSlopes = np.asarray(slopePositiveLines)[:,4]
    posSlopeMedian = np.median(positiveSlopes)
    posSlopeStdDev = np.std(positiveSlopes)
    posSlopesGood = []
    for slope in positiveSlopes:
        if abs(slope-posSlopeMedian) < .9:
            posSlopesGood.append(slope)
    posSlopeMean = np.mean(np.asarray(posSlopesGood))
            
    
    negativeSlopes = np.asarray(slopeNegativeLines)[:,4]
    negSlopeMedian = np.median(negativeSlopes)
    negSlopeStdDev = np.std(negativeSlopes)
    negSlopesGood = []
    for slope in negativeSlopes:
        if abs(slope-negSlopeMedian) < .9:
            negSlopesGood.append(slope)
    negSlopeMean = np.mean(np.asarray(negSlopesGood))
        
    #--------------------------Get Average x Coord When y Coord Of Line = 0----------------------------
    # Positive Lines
    xInterceptPos = []
    for line in slopePositiveLines:
        x1 = line[0]
        y1 = im.shape[0]-line[1]
        slope = line[4]
        yIntercept = y1-slope*x1
        xIntercept = -yIntercept/slope
        xInterceptPos.append(xIntercept)
    xInterceptPosMean = np.mean(np.asarray(xInterceptPos))
    
    # Negative Lines 
    xInterceptNeg = []
    for line in slopeNegativeLines:
        x1 = line[0]
        y1 = im.shape[0]-line[1]
        slope = line[4]
        yIntercept = y1-slope*x1
        xIntercept = -yIntercept/slope
        xInterceptNeg.append(xIntercept)
    xInterceptNegMean = np.mean(np.asarray(xInterceptNeg))
    
    # ----------------------PLOT LANE LINES------------------------------
    # Need end points of line to draw in. Have x1,y1 (xIntercept,im.shape[1]) where
    # im.shape[1] is the bottom of the image. take y2 as some num (min/max y in the good lines?)
    # then find corresponding x
    
    # make new black image
    laneLines = np.zeros_like(edgesIm)   
    colorLines = im.copy()
    
    # Positive Slope Line
    slope = posSlopeMean
    x1 = xInterceptPosMean
    y1 = 0
    y2 = imshape[0] - (imshape[0]-imshape[0]*.35)
    x2 = (y2-y1)/slope + x1
    
    # Plot positive slope line
    x1 = int(round(x1))
    x2 = int(round(x2))
    y1 = int(round(y1))
    y2 = int(round(y2))
    cv2.line(laneLines,(x1,im.shape[0]-y1),(x2,imshape[0]-y2),(255,255,0),2) # plot line
    cv2.line(colorLines,(x1,im.shape[0]-y1),(x2,imshape[0]-y2),(0,255,0),4) # plot line on color image
    
    # Negative Slope Line
    slope = negSlopeMean
    x1N = xInterceptNegMean
    y1N = 0
    x2N = (y2-y1N)/slope + x1N
    
    # Plot negative Slope Line
    x1N = int(round(x1N))
    x2N = int(round(x2N))
    y1N = int(round(y1N))
    cv2.line(laneLines,(x1N,imshape[0]-y1N),(x2N,imshape[0]-y2),(255,255,0),2)
    cv2.line(colorLines,(x1N,im.shape[0]-y1N),(x2N,imshape[0]-y2),(0,255,0),4) # plot line on color iamge
    
    # Plot lane lines
    #plt.figure(8)
    #plt.imshow(laneLines,cmap='gray')
    #plt.title('Lane Lines')
    
    # Plot lane lines on original image
    #plt.figure(9)
    #plt.imshow(colorLines)
    #plt.title('Lane Lines Color Image')
    
    
    #-------------------------------------Blend Image-----------------------------------------
    laneFill = im.copy()
    vertices = np.array([[(x1,im.shape[0]-y1),(x2,im.shape[0]-y2),  (x2N,imshape[0]-y2),
                                          (x1N,imshape[0]-y1N)]], dtype=np.int32)
    color = [241,255,1]
    cv2.fillPoly(laneFill, vertices, color)
    opacity = .1
    blendedIm =cv2.addWeighted(laneFill,opacity,im,1-opacity,0,im)
    cv2.line(blendedIm,(x1,im.shape[0]-y1),(x2,imshape[0]-y2),(0,255,0),4) # plot line on color image
    cv2.line(blendedIm,(x1N,im.shape[0]-y1N),(x2N,imshape[0]-y2),(0,255,0),4) # plot line on color image
    
    # Plot final output
    #plt.figure(10)
    #plt.imshow(blendedIm)
    #plt.title('Final Output')
    
    
    # write the frame
    videoOut.write(blendedIm)

# Release video
videoOut.release()