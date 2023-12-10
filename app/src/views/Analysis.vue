<template>
  <div class="camera-container">
    <video ref="videoElement" class="camera-preview" autoplay></video>
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

    this.socket.addEventListener('message', (event) => {
      // 处理从服务器接收到的图像数据
      const imageData = event.data;

      console.log('Received image data:', imageData);

      // 在网页上显示图像，这里假设服务器发送的是JPEG格式的图像
      const blob = new Blob([imageData], { type: 'image/jpeg' });
      const imageUrl = URL.createObjectURL(blob);

      this.$refs.videoElement.src = imageUrl;
    });
  },
  methods: {
    connectWebSocket() {
      this.socket = new WebSocket('ws://localhost:9002');
    }
  }
};
</script>

<style scoped>
.camera-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.camera-preview {
  width: 100%;
  max-width: 640px;
  height: auto;
}
</style>

  