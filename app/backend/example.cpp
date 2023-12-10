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
        try {
            // Read image file
            std::string message = msg->get_payload();
            cout << "Received message from client: " << message << endl;
            cv::Mat image = cv::imread("../../imgs/image.jpg", cv::IMREAD_COLOR);

            // Convert OpenCV Mat to a byte vector
            vector<uchar> buffer;
            cv::imencode(".jpg", image, buffer);

            // Send the image data as a string to the client
            server.send(hdl, buffer.data(), buffer.size(), websocketpp::frame::opcode::binary);
        } catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
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

