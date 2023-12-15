#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

class Camera {
public:
    Camera(int cameraIndex = 0);
    ~Camera();

    bool openCamera();
    void closeCamera();

    void captureAndShow();
    cv::Mat getFrame();  // New function to get the current frame

private:
    cv::VideoCapture cap;
    cv::Mat frame;
    std::string windowName;
};

#endif // CAMERA_H
