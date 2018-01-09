# Lane Detection in OpenCV
![gif](https://user-images.githubusercontent.com/25371934/33930574-61018b38-dfbb-11e7-89bb-66bea1bbb021.gif)

## Overview
Lane detection for a car driving on the highway using OpenCV in python. Two videos are used and the detected lane is drawn over each frame. The output videos correspond to the test_videos. 

## Method
For each frame, the first step is to grayscale the image

![original](https://user-images.githubusercontent.com/25371934/34698298-651b5af6-f4a5-11e7-92f7-2eb99d7603b1.png)


![grayscaled](https://user-images.githubusercontent.com/25371934/34698324-906ba350-f4a5-11e7-9a25-b62b42825aba.png)


then a gaussian filter is applied 

![gaussiansmoothed](https://user-images.githubusercontent.com/25371934/34701780-1a4d5e76-f4b8-11e7-8b85-9814488b5b42.png)

Canny edge detection is used on the smoothed image

![cannyedge](https://user-images.githubusercontent.com/25371934/34701790-254b3a5a-f4b8-11e7-9bc6-35349f6949e8.png)

An area of interest is used to filter out irrelevant parts of the image

![masked](https://user-images.githubusercontent.com/25371934/34701805-3a5600e2-f4b8-11e7-8af9-9ec665bd4f79.png)

Hough lines are found and separated into positive and negative slopes

![houghlinesall](https://user-images.githubusercontent.com/25371934/34701814-4851d932-f4b8-11e7-92de-ca63f5782ce3.png)

Slope averages and x intercept averages are found for positive and negative lines, and the average lane line for positive and negative are found

![lanelinesaverage](https://user-images.githubusercontent.com/25371934/34701832-56d626f2-f4b8-11e7-9210-c213e992d6f8.png)

The lane is filled in and the result is written to the output video

![finalout](https://user-images.githubusercontent.com/25371934/34701845-6edce786-f4b8-11e7-8428-7ad5690e160e.png)

## Dependencies
- numpy
- matplotlib
- OpenCV 3.x

## Running
Running LaneDetect_OneFrame only uses one frame and displays images at each step of the process. Running LaneDetect will go through the frames in test_video and write an output.avi. The test video used is currently specified on lines 19 and 20.

## TODO
Use Homography to get birds eye view of image, fit a curve to the lane in this view and transform back to draw lane lines more accurately.

## References
- https://drive.google.com/open?id=0B8DbLKogb5ktZEpQeGJiNV9pWDA
- https://drive.google.com/open?id=0B8DbLKogb5ktcGEzVDBhWlp5SG8
- https://github.com/jessicayung/self-driving-car-nd

