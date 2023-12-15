#ifndef WEBSOCKETSERVER_H
#define WEBSOCKETSERVER_H

#include "ModelPredict.h"

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <websocketpp/common/thread.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

class WebSocketServer {
public:
    typedef websocketpp::server<websocketpp::config::asio> Server;
    typedef Server::message_ptr MessagePtr;

    WebSocketServer();

    void on_message(websocketpp::connection_hdl hdl, MessagePtr msg);
    void on_open(websocketpp::connection_hdl hdl);
    void on_close(websocketpp::connection_hdl hdl);
    void start(uint16_t port);

private:
    char* model_path_;
    ModelPredict onnx_mp_;
    Server server_;
};

#endif // WEBSOCKETSERVER_H
