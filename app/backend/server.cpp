#include "Camera.h"

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <websocketpp/common/thread.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

using namespace std;

class WebSocketServer {
public:
    typedef websocketpp::server<websocketpp::config::asio> Server;
    typedef Server::message_ptr MessagePtr;

    WebSocketServer() {
        server.init_asio();

        server.set_message_handler(bind(&WebSocketServer::on_message, this, placeholders::_1, placeholders::_2));
        server.set_open_handler(bind(&WebSocketServer::on_open, this, placeholders::_1));
        server.set_close_handler(bind(&WebSocketServer::on_close, this, placeholders::_1));
    }

    void on_message(websocketpp::connection_hdl hdl, MessagePtr msg) {
        std::string message;
        try {
            message = msg->get_payload();
            cout << "Received message from client: " << message << endl;
            // cv::Mat image = cv::imread("../../imgs/image.jpg", cv::IMREAD_COLOR);
        } catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
            return;
        }

        if (message == "OpenCamera"){
            
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

                    // Convert OpenCV Mat to a byte vector
                    vector<uchar> buffer;
                    std::cout << "Current frame type is:" << currentFrame.type() << std::endl;
                    std::cout << "Current frame channels is:" << currentFrame.channels() << std::endl;
                    currentFrame.convertTo(convertedFrame, CV_8U);
                    cv::imencode(".jpg", convertedFrame, buffer);

                    if (buffer.empty()) {
                        std::cout << "Error: Buffer is empty, no image data." << std::endl;
                    } else {
                        std::cout << "Buffer contains image data." << std::endl;
                    }

                    // Send the image data as a string to the client
                    server.send(hdl, buffer.data(), buffer.size(), websocketpp::frame::opcode::binary);

                    // Add a delay to control the frame rate
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }).detach(); // Detach the thread so it runs in the background
        }
    }

    void on_open(websocketpp::connection_hdl hdl) {
        cout << "Client connected" << endl;
    }

    void on_close(websocketpp::connection_hdl hdl) {
        cout << "Client disconnected" << endl;
    }

    void start(uint16_t port) {
        try {
            server.listen(port);
            server.start_accept();
        
            cout << "Server listening on " << "localhost:" << port << endl;

            server.run();
        } catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
        }
    }

private:
    Server server;
};

int main() {
    WebSocketServer wsServer;
    wsServer.start(9002); // 选择一个未被占用的端口
    return 0;
}

