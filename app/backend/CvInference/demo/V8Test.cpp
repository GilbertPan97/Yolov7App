
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "inference.h"
#include "../../Camera.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    bool runOnGPU = false;

    // Set input onnx model
    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf("../../yolov8.onnx", cv::Size(640, 480), "classes.txt", runOnGPU);

    Camera sensor;
    bool staCam = sensor.openCamera();

    while (staCam == true)
    {
        cv::Mat frame = sensor.getFrame();

        cv::Mat frame_inf = frame.clone();

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        // Note: Need resize before inference
        // cv::resize(frame, frame, cv::Size(480, 640));

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame_inf, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame_inf, textBox, color, cv::FILLED);
            cv::putText(frame_inf, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }

        // Conbined images: original image and inference image
        cv::Mat disp_img;
        cv::Mat separator(frame.rows, 5, CV_8UC3, cv::Scalar(255));
        cv::hconcat(frame, separator, frame);         // add separate line to original image
        cv::hconcat(frame, frame_inf, disp_img);

        // Display original and result image with opencv
        cv::namedWindow("Original and inference image", cv::WINDOW_NORMAL);
        cv::imshow("Original and inference image", disp_img);

        cv::waitKey(50);
    }
}

