#include "WebSocketServer.h"
#include "Camera.h"

#include <iostream>

using namespace std;

WebSocketServer::WebSocketServer() : inf_("./models/yolov8.onnx", cv::Size(640, 480), "classes.txt", true){

    server_.init_asio();

    server_.set_message_handler(bind(&WebSocketServer::on_message, this, placeholders::_1, placeholders::_2));
    server_.set_open_handler(bind(&WebSocketServer::on_open, this, placeholders::_1));
    server_.set_close_handler(bind(&WebSocketServer::on_close, this, placeholders::_1));
}

void WebSocketServer::on_message(websocketpp::connection_hdl hdl, MessagePtr msg) {
    std::string message;
    try {
        message = msg->get_payload();
        cout << "Received message from client: " << message << endl;
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return;
    }

    if (message == "OpenCamera") {
        // Set up a timer to periodically send images
        std::thread([&, hdl]() {
            Camera camera(0);
            if (!camera.openCamera()) {
                std::cerr << "Error: Fail to open camera." << std::endl;
                return;
            }

            cv::Mat currentFrame;
            cv::Mat convertedFrame;
            while (true) {
                currentFrame = camera.getFrame();

                // Check if the frame is empty
                if (currentFrame.empty()) {
                    cerr << "Error: Empty frame received from the camera." << endl;
                    break;
                }

                // Run inference
                std::vector<Detection> output = inf_.runInference(currentFrame);
                cv::Mat frame_inf = currentFrame.clone();
                for (int i = 0; i < output.size(); ++i)
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

                // Convert OpenCV Mat to a byte vector
                std::vector<uchar> buffer;
                frame_inf.convertTo(convertedFrame, CV_8U);
                cv::imencode(".jpg", convertedFrame, buffer);

                if (buffer.empty()) {
                    std::cout << "Error: Buffer is empty, no image data." << std::endl;
                } else {
                    std::cout << "Buffer contains image data." << std::endl;
                }

                // Send the image data as a string to the client
                server_.send(hdl, buffer.data(), buffer.size(), websocketpp::frame::opcode::binary);

                // Add a delay to control the frame rate
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }).detach(); // Detach the thread so it runs in the background
    }
}

void WebSocketServer::on_open(websocketpp::connection_hdl hdl) {
    cout << "Client connected" << endl;
}

void WebSocketServer::on_close(websocketpp::connection_hdl hdl) {
    cout << "Client disconnected" << endl;
}

void WebSocketServer::start(uint16_t port) {
    try {
        server_.listen(port);
        server_.start_accept();
    
        cout << "Server listening on " << "localhost:" << port << endl;

        server_.run();
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}
