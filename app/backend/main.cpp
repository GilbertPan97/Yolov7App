#include "WebSocketServer.h"

#include <iostream>

int main() {
    WebSocketServer wsServer;
    wsServer.start(9002);       // open a port which not used
    
    return 0;
}

