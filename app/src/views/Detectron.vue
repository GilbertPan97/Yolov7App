<template>
  <div>
    <div class="viewerContainer">
      <img ref="viewerElement" class="viewer" />
      <!-- <video ref="viewerElement" class="viewer" autoplay></video> -->
    </div>
    <div class="widgetContainer">
      <button class="button" @click="openCamera">Open Camera</button>
      <button class="button" @click="closeCamera">Close Camera</button>
      <button class="button" @click="runYolo">Run Yolo</button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      socket: null
    };
  },
  mounted() {
    this.connectWebSocket();

    this.socket.addEventListener('open', (event) => {
      console.log('WebSocket connection opened:', event);
    });

    this.socket.addEventListener('error', (event) => {
      console.error('WebSocket error:', event);
    });

    this.socket.addEventListener('close', (event) => {
      console.log('WebSocket connection closed:', event);
    });

    this.socket.addEventListener('message', (event) => {
      // 处理从服务器接收到的图像数据
      const imageData = event.data;

      console.log('Received image data:', imageData);

      // 在网页上显示图像，这里假设服务器发送的是JPEG格式的图像
      const blob = new Blob([imageData], { type: 'image/jpeg' });
      const imageUrl = URL.createObjectURL(blob);

      // Assuming `dispElement` is a reference to your img element
      this.$refs.viewerElement.src = imageUrl;

      // You may not need the loadedmetadata event for an img element
    });
  },

  methods: {
    connectWebSocket() {
      this.socket = new WebSocket('ws://localhost:9002');
    },
    openCamera() {
      // Send message when button pressed.
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.send('OpenCamera');
      } else {
        console.error('WebSocket connect is not opened.');
      }
    }
  }
};
</script>

<style scoped>
@import "../assets/style.css";
.viewerContainer {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.widgetContainer {
  margin: 5px;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
}

.button {
  margin: 5px;
}

.viewer {
  width: 100%;
  max-width: 640px;
  height: auto;
  border-radius: 8px;
  border: 2px solid;
  object-fit: contain;
}

</style>
