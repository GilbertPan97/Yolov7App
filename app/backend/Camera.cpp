#include "Camera.h"

Camera::Camera(int cameraIndex) {
    cap = cv::VideoCapture(cameraIndex);
    windowName = "Camera View";
}

Camera::~Camera() {
    closeCamera();
}

bool Camera::openCamera() {
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera." << std::endl;
        return false;
    }
    return true;
}

void Camera::closeCamera() {
    if (cap.isOpened()) {
        cap.release();
        cv::destroyAllWindows();
    }
}

void Camera::captureAndShow() {
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame;

        if (!frame.data) {
            std::cerr << "Error: Unable to capture frame. Exiting." << std::endl;
            break;
        }

        cv::imshow(windowName, frame);
        int key = cv::waitKey(20);

        if (key == 27 || key == 'Q' || key == 'q') {
            break;
        }
    }

    closeCamera();
}

cv::Mat Camera::getFrame() {
	cap >> frame;
    return frame.clone();  // Return a copy of the current frame
}
