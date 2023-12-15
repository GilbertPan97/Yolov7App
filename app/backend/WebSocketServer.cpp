#include "WebSocketServer.h"
#include "Camera.h"

#include <iostream>

using namespace std;

WebSocketServer::WebSocketServer() : onnx_mp_(true, 0){

    model_path_ = "../../models/mask_rcnn_sim.onnx";
    try {
        bool sta = onnx_mp_.LoadModel(model_path_);
        cout << "Info: Load model status: " << sta << endl;
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return;
    }

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

                // Onnx Predict
                float score_thresh = 0.6f;
		        bool status = onnx_mp_.PredictAction(currentFrame, score_thresh);
                cv::Mat result_img = onnx_mp_.ShowPredictMask(currentFrame, score_thresh);

                // Convert OpenCV Mat to a byte vector
                std::vector<uchar> buffer;
                result_img.convertTo(convertedFrame, CV_8U);
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
